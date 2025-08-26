# AI and ML exercises-  RAG Chatbot Demo

We will be creating an AI assistant you can interact with to ask AMD related questions.

First, create a directory for the necessary software and the software dependencies:

```
mkdir ai_assistant_install
cd ai_assistant_install
export AI_ASSISTANT_INSTALL_DIR=$PWD
cd ..
mkdir ai_assistant_dependencies
cd ai_assistant_dependencies
export AI_ASSISTANT_DEPS_DIR=$PWD
cd ..
```

Then, clone our exercises repo and move to the relevant directory:

```
git clone https://github.com/amd/HPCTrainingExamples.git
cd HPCTrainingExamples/MLExamples/RAG_LangChainDemo
```



Once in the above directory, you will see a file called `requirements.txt` which we will be using to install the requirements for LangChain:

```
pip3 install -r requirements.txt 

Throughout these exercises we'll be leveraging the existing ROCm installation (minimum 6.2). 
It is expected that ROCm installation has been completed prior to installing the dependencies outlined below.


Also, note that these exercises are prepared for MI300 GPU series.


## Running the LLM Server
For this demo, we will be using Ollama to serve Llama3.1:70b.    Instructions for installing Ollama can be found on Ollama website.
Once installed, use this command to launch the LLM:
ollama run llama3.1:70b

```


```
## Installing the required packages
On command line in same directory where the included requirements.txt file is located, run the following command:

pip install –r requirements.txt

```
```
## Running the script
On command line in same directory where the included instinct_chat.py file is located, run the following command:

python3 ./instinct_chat.py

```
```
## Querying the chatbot
After the script launch is complete, a URL should be displayed in the terminal window.
In a local web browser, navigate to that URL.

You can now enter your question for the ROCm chatbot and it will return answers on the ROCm product.


