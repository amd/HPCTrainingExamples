# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved This software is distributed under the MIT License, Contact: Peter Cross
import requests
import logging
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from bs4 import BeautifulSoup
import gradio as gr
from urllib.parse import urljoin

# Function to scrape content from a URL
def scrape_url(url):
    try:
        response = requests.get(url, timeout=3)
        
        # Check if the content is HTML
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            print(f"Skipping non-HTML content at {url}")
            return ""

        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(strip=True)
    except requests.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return ""

# Main website URL
main_url = "https://rocm.docs.amd.com/en/latest/"

# Scrape the main page
try:
    main_page = requests.get(main_url)
    main_soup = BeautifulSoup(main_page.text, 'html.parser')
except requests.RequestException as e:
    print(f"Error accessing main page: {e}")
    exit(1)

# Find all links on the main page
links = main_soup.find_all('a', href=True)

# Scrape content from the main page and linked pages
all_content = []
all_content.append(Document(page_content=scrape_url(main_url), metadata={"source": main_url}))
for link in links:
    full_url = urljoin(main_url, link['href'])
    if (full_url == "https://www.amd.com/"):
        print("detected main amd site-ignoring")
    elif (full_url == "https://www.amd.com/en/developer/resources/infinity-hub.html"):
        print("detected Infinity Hub-ignoring")
    elif full_url.startswith('http') and not full_url.lower().endswith('.pdf'):
        content = scrape_url(full_url)
        print(f"Scraping site: ", full_url)
        if content:
            all_content.append(Document(page_content=content, metadata={"source": full_url}))

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(all_content)

# Create embeddings
embeddings = HuggingFaceEmbeddings()

# Create a vector store
vectorstore = Chroma.from_documents(texts, embeddings)

# Create a retriever
retriever = vectorstore.as_retriever()

# Define four LLMs
llm1 = Ollama(model="llama3:70b")
llm2 = Ollama(model="gemma2:27b")
llm3 = Ollama(model="mistral-large")  # Add a third model
llm4 = Ollama(model="phi3:14b")    # Add a fourth model

# Define the prompt
prompt = """
1. Use the following pieces of context to answer the question related to AMD's ROCm product.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
3. Keep the answer crisp and limited to 3,4 sentences, and try to only use the context provided when answering.

Context: {context}

Question: {question}

Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt) 

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

# Function to create QA chain
def create_qa_chain(llm):
    llm_chain = LLMChain(
        llm=llm, 
        prompt=QA_CHAIN_PROMPT, 
        callbacks=None, 
        verbose=True
    )
    
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        callbacks=None
    )
    
    return RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        verbose=True,
        retriever=retriever,
        return_source_documents=True
    )

# Create four QA chains
qa1 = create_qa_chain(llm1)
qa2 = create_qa_chain(llm2)
qa3 = create_qa_chain(llm3)
qa4 = create_qa_chain(llm4)

# Function to get responses from all four LLMs
def respond(question, history):
    response1 = qa1(question)["result"]
    response2 = qa2(question)["result"]
    response3 = qa3(question)["result"]
    response4 = qa4(question)["result"]
    return response1, response2, response3, response4

# Create Gradio interface
def create_interface():
    with gr.Blocks(theme="glass") as demo:
        gr.Markdown("# Rocm-bot (Quad LLM)")
        
        chatbot1 = gr.Chatbot(label="LLM1 (llama3:70b)", height=300)
        chatbot2 = gr.Chatbot(label="LLM2 (gemma2:27b)", height=300)
        chatbot3 = gr.Chatbot(label="LLM3 (mistral-large)", height=300)
        chatbot4 = gr.Chatbot(label="LLM4 (phi3:14b)", height=300)
        
        msg = gr.Textbox(
            placeholder="Ask me questions related to the awesomeness of ROCm and how it can revolutionize your AI workflows",
            container=False,
            scale=7
        )

        def user(user_message, history1, history2, history3, history4):
            response1, response2, response3, response4 = respond(user_message, None)
            history1.append((user_message, response1))
            history2.append((user_message, response2))
            history3.append((user_message, response3))
            history4.append((user_message, response4))
            return "", history1, history2, history3, history4

        msg.submit(user, [msg, chatbot1, chatbot2, chatbot3, chatbot4], [msg, chatbot1, chatbot2, chatbot3, chatbot4])

        gr.Examples(
            examples=["How can I install ROCm", "What installation methods exist for ROCm"],
            inputs=msg,
        )

    return demo

# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0")
    #demo.launch(auth=("admin", "rocm4th@win"), share=True, server_name="0.0.0.0")