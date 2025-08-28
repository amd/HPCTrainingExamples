import asyncio
import aiohttp
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from io import BytesIO
import tempfile
import time
import os
import sys
import pickle

from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# =========================
# Configurable variables
# =========================
TIMEOUT = 5  # seconds
MAX_DEPTH = 1
CRAWL_DELAY = 1  # seconds delay between requests to avoid overload
CONCURRENT_REQUESTS = 5  # Limit max concurrent requests for politeness

visited_urls = set()
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

def should_ignore_url(url):
    allowed_amd_subdomains = [
        "rocm.docs.amd.com",
        "rocm.blogs.amd.com"
    ]
    if "amd.com" in url:
        if not any(subdomain in url for subdomain in allowed_amd_subdomains):
            return True
    return False

def scrape_pdf_from_bytes(pdf_bytes):
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_bytes.getbuffer())
        temp_pdf.flush()
        loader = PDFPlumberLoader(temp_pdf.name)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])
    return full_text

async def fetch(session, url):
    try:
        async with semaphore:
            async with session.get(url, timeout=ClientTimeout(total=TIMEOUT)) as response:
                content_type = response.headers.get('Content-Type', '').lower()
                if response.status != 200:
                    print(f"Failed to fetch {url}: HTTP {response.status}")
                    return url, None, content_type
                content = await response.read()
                return url, content, content_type
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return url, None, None

async def scrape_url_or_pdf(session, url):
    url, content, content_type = await fetch(session, url)
    if content is None:
        return ""

    if url.lower().endswith('.pdf') or 'application/pdf' in content_type:
        print(f"Scraping PDF: {url}")
        pdf_bytes = BytesIO(content)
        return scrape_pdf_from_bytes(pdf_bytes)

    elif 'text/html' in content_type:
        text = content.decode(errors='ignore')
        soup = BeautifulSoup(text, 'html.parser')

        if 'rocm.blogs.amd.com' in url:
            slide_texts = []
            tasks = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if href.lower().endswith('.pdf'):
                    slide_url = urljoin(url, href)
                    if slide_url not in visited_urls and not should_ignore_url(slide_url):
                        visited_urls.add(slide_url)
                        tasks.append(scrape_url_or_pdf(session, slide_url))
            if tasks:
                results = await asyncio.gather(*tasks)
                slide_texts.extend([r for r in results if r])
            main_text = soup.get_text(separator='\n', strip=True)
            return main_text + "\n" + "\n".join(slide_texts)

        return soup.get_text(separator='\n', strip=True)

    else:
        print(f"Skipping unsupported content type at {url}")
        return ""

async def recursive_scrape(session, url, base_url, max_depth=MAX_DEPTH, current_depth=0):
    if current_depth > max_depth:
        return []

    if url in visited_urls:
        return []

    if should_ignore_url(url):
        print(f"Ignoring URL (filtered amd.com): {url}")
        return []

    visited_urls.add(url)
    print(f"Scraping (depth {current_depth}): {url}")

    documents = []
    content = await scrape_url_or_pdf(session, url)
    if content:
        documents.append(Document(page_content=content, metadata={"source": url}))

    try:
        _, raw_content, content_type = await fetch(session, url)
        if raw_content and 'text/html' in content_type:
            text = raw_content.decode(errors='ignore')
            soup = BeautifulSoup(text, 'html.parser')
            links = soup.find_all('a', href=True)

            tasks = []
            for link in links:
                full_url = urljoin(base_url, link['href'])
                if full_url.startswith('http') and full_url not in visited_urls:
                    if base_url in full_url and not should_ignore_url(full_url):
                        await asyncio.sleep(CRAWL_DELAY)
                        tasks.append(recursive_scrape(session, full_url, base_url, max_depth, current_depth + 1))

            if tasks:
                results = await asyncio.gather(*tasks)
                for result_docs in results:
                    documents.extend(result_docs)
    except Exception as e:
        print(f"Error accessing {url} for recursion: {e}")

    return documents


async def scrape_all(rocm_version):
    main_urls = [
        rocm_docs_url=rocm_version
        if rocm_version == "latest":
           rocm_docs_url=rocm_version
        else:
           rocm_docs_url=f"docs-{rocm_version}"
        f"https://rocm.docs.amd.com/en/{rocm_docs_url}/",
        "https://rocm.blogs.amd.com/verticals-ai.html",
        "https://rocm.blogs.amd.com/verticals-ai-page2.html",
        "https://rocm.blogs.amd.com/verticals-ai-page3.html",
        "https://rocm.blogs.amd.com/verticals-ai-page4.html",
        "https://rocm.blogs.amd.com/verticals-ai-page5.html",
        "https://rocm.blogs.amd.com/verticals-ai-page6.html",
        "https://rocm.blogs.amd.com/verticals-ai-page7.html",
        "https://rocm.blogs.amd.com/verticals-ai-page8.html",
        "https://rocm.blogs.amd.com/verticals-ai-page9.html",
        "https://rocm.blogs.amd.com/verticals-ai-page10.html",
        "https://rocm.blogs.amd.com/verticals-ai-page11.html",
        "https://rocm.blogs.amd.com/verticals-ai-page12.html",
        "https://rocm.blogs.amd.com/verticals-ai-page13.html",
        "https://rocm.blogs.amd.com/verticals-ai-page14.html",
        "https://rocm.blogs.amd.com/verticals-hpc.html",
        "https://rocm.blogs.amd.com/verticals-hpc-page2.html",
        "https://rocm.blogs.amd.com/verticals-hpc-page3.html",
        "https://rocm.blogs.amd.com/verticals-hpc-page4.html",
        "https://rocm.blogs.amd.com/hpc-ecosystems-and-partners.html",
        "https://rocm.blogs.amd.com/hpc-applications-and-models.html",
        "https://rocm.blogs.amd.com/hpc-applications-and-models-page2.html,"
        "https://rocm.blogs.amd.com/hpc-software-tools-and-optimizations.html",
        "https://rocm.blogs.amd.com/hpc-software-tools-and-optimizations-page2.html,"
        "https://rocm.blogs.amd.com/hpc-software-tools-and-optimizations-page3.html,"
        "https://rocm.blogs.amd.com/verticals-developers.html",
        "https://rocm.blogs.amd.com/verticals-developers-page2.html",
        "https://rocm.blogs.amd.com/verticals-developers-page3.html",
        "https://rocm.blogs.amd.com/software-tools-optimizations.html",
        "https://rocm.blogs.amd.com/software-tools-optimizations-page2.html",
        "https://rocm.blogs.amd.com/software-tools-optimizations-page3.html",
        "https://rocm.blogs.amd.com/software-tools-optimizations-page4.html",
        "https://rocm.blogs.amd.com/software-tools-optimizations-page5.html",
        "https://rocm.blogs.amd.com/ai-software-tools-and-optimizations.html",
        "https://rocm.blogs.amd.com/ai-software-tools-and-optimizations-page2.html",
        "https://rocm.blogs.amd.com/ai-software-tools-and-optimizations-page3.html",
        "https://rocm.blogs.amd.com/ai-applications-and-models.html",
        "https://rocm.blogs.amd.com/ai-applications-and-models-page2.html",
        "https://rocm.blogs.amd.com/ai-applications-and-models-page3.html",
        "https://rocm.blogs.amd.com/ai-applications-and-models-page4.html",
        "https://rocm.blogs.amd.com/ai-applications-and-models-page5.html",
        "https://rocm.blogs.amd.com/ai-applications-and-models-page6.html",
        "https://rocm.blogs.amd.com/ai-applications-and-models-page7.html",
        "https://rocm.blogs.amd.com/ai-applications-and-models-page8.html",
        "https://rocm.blogs.amd.com/ai-applications-and-models-page9.html",
        "https://rocm.blogs.amd.com/ai-applications-and-models-page10.html",
        "https://rocm.blogs.amd.com/ai-applications-and-models-page11.html",
        "https://rocm.blogs.amd.com/ai-ecosystems-and-partners.html",
        "https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html",
        "https://rocm.blogs.amd.com/verticals-systems.html",
        "https://rocm.blogs.amd.com/verticals-systems-page2.html",
        "https://rocm.blogs.amd.com/verticals-data-science.html",
        "https://rocm.blogs.amd.com/verticals-data-science-page2.html",
        "https://rocm.blogs.amd.com/verticals-data-science-page3.html",
        "https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-readme/",
        "https://fs.hlrs.de/projects/par/events/2025/GPU-AMD/",
        "https://github.com/amd/HPCTrainingExamples/blob/main/README.md",
        "https://github.com/amd/HPCTrainingDock/blob/main/README.md"
    ]

    all_content = []
    async with aiohttp.ClientSession() as session:
        tasks = [recursive_scrape(session, url, url, max_depth=MAX_DEPTH) for url in main_urls]
        results = await asyncio.gather(*tasks)
        for docs in results:
            all_content.extend(docs)
    return all_content


def save_docs(all_content, path="scraped_docs.pkl"):
    with open(path, "wb") as f:
        pickle.dump(all_content, f)

def load_docs(path="scraped_docs.pkl"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


async def main():
    persist_dir = "chroma_store"
    docs_path = "scraped_docs.pkl"

    # Default values
    force_scrape = False
    rocm_version = "latest"

    # Parse args manually for simplicity
    for i, arg in enumerate(sys.argv):
        if arg == "--scrape":
            force_scrape = True
        if arg == "--rocm-version" and i + 1 < len(sys.argv):
            rocm_version = sys.argv[i + 1]

    print(f"Using ROCm version: {rocm_version}")
    print(f"Force scrape: {force_scrape}")

    # Load scraped docs or scrape if none or forced
    all_content = None
    if not force_scrape:
        all_content = load_docs(docs_path)
        if all_content:
            print(f"Loaded {len(all_content)} documents from disk.")
        else:
            print("No saved documents found, scraping now...")
    if force_scrape or not all_content:
        all_content = await scrape_all(rocm_version)
        print(f"Scraped {len(all_content)} documents.")
        save_docs(all_content, docs_path)

    embeddings = HuggingFaceEmbeddings()

    # Load or create vectorstore
    if os.path.exists(persist_dir) and not force_scrape:
        print("Loading existing vectorstore...")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        print("Creating new vectorstore and persisting to disk...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(all_content)
       # vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=persist_dir)
        vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_dir)

        # Batch insertion
        BATCH_SIZE = 5000  # Use a safe value below the limit

        for i in range(0, len(texts), BATCH_SIZE):
           batch = texts[i:i+BATCH_SIZE]
           vectorstore.add_documents(batch)

        vectorstore.persist()

    retriever = vectorstore.as_retriever()

    prompt = """
    1. Use the following pieces of context to answer the question related to AMD's products and software.
    2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.
    3. Keep the answer crisp and limited to 5,6 sentences, and try to only use the context provided when answering.

    Context: {context}

    Question: {question}

    Helpful Answer:
    """

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    llm = Ollama(model="llama3.3:70b")

    llm_chain = LLMChain(
        llm=llm,
        prompt=QA_CHAIN_PROMPT,
        callbacks=None,
        verbose=True
    )

    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        callbacks=None
    )

    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        verbose=True,
        retriever=retriever,
        return_source_documents=True
    )

    def respond(question, history=None):
        return qa(question)["result"]

    print("\nAMD AI Assistant Ready! Type your questions. Type 'exit', 'quit' or 'bye' to stop.\n")

    while True:
        user_input = input("Prompt: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        answer = respond(user_input, history=None)
        print(f"AMD AI Assistant: {answer}\n")


if __name__ == "__main__":
    asyncio.run(main())

