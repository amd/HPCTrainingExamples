# AI and ML exercises-  RAG Chatbot Demo

**NOTE**: these exercises have been tested on MI300X accelerators.
To see details on the environment (such as operating system and modules available) please see `README.md` on [this](https://github.com/amd/HPCTrainingDock) repo.

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


