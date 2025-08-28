# AI and ML exercises-  RAG Chatbot Demo

We will be creating an AI assistant you can interact with to ask AMD related questions.

First, create a directory for the necessary software and the software dependencies:

```
mkdir ai_assistant_deps
cd ai_assistant_deps
export AI_ASSISTANT_DEPS=$PWD
cd ..
```

Then, clone our exercises repo and move to the relevant directory:

```
git clone https://github.com/amd/HPCTrainingExamples.git
cd HPCTrainingExamples/MLExamples/RAG_LangChainDemo
```

Once in the above directory, you will see a file called `requirements.txt` which we will be using to install the software requirements for the AI assistant:

```
pip3 install -r requirements.txt --target=$AI_ASSISTANT_DEPS
```

Make sure to export the paths of the software just installed into your `PYTHONPATH`:

```
export PYTHONPATH=$AI_ASSISTANT_DEPS:$PYTHONPATH
```

Running the AI assistant needs the Ollama server to run the LLM model on which it is based: to install Ollama see these [instructions](https://ollama.com/download).

For this example, we are going to be using `Llama3.3:70b`. To setup ollama after installation, run the following commands. Note that the first command below will kill all ollama processes already running on your system
```
pkill -f ollama
ollama serve &
ollama pull llama3.3:70b
```
To test that ollama is working you can do:
```
ollama run llama3.3:70b
```
The above command will run the LLM  locally, you can interact with it through the prompt and then exit with `/bye`.

There currently are three versions of the AI assistant script you can run:

1. instinct_chat.py
2. instinct_chat_4_llm.py
3. amd_ai_assistant.py

All the above scripts will implement a retrieval augmented generation (RAG) model by scraping the web to get information on AMD and AMD software to provide the necessary context to Llama3.3 to be able to answer AMD specific questions leveraging the info from those specific websites.

## Running the scripts
All scripts can be run with `python3` as follows:

```
python3 ./instinct_chat.py
python3 ./instinct_chat_4_llm.py
python3 ./amd_ai_assistant.py

```

Note that the first two scripts will use a web interface to interact with the model, whereas the third one will work on the command line locally.
To use the web interface, copy the URL that is displayed and paste it in your browser.
The `amd_ai_assistant.py` script will save the scraped pages in a file and will not scrape again if you run it a second time, unless you provide the `--scrape` flag when launching it. You can also provide the ROCm version with `--rocm-version <version>` to scrape from the documentation associated to the input version, otherwise all scripts will scrape from the latests docs.


