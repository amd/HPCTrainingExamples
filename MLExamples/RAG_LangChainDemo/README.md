# AMD AI Assistant using retrieval augmented generation (RAG) 

We will be using retrieval augmented generation (RAG) to create an AMD AI assistant you can interact with to answer questions on AMD GPU software and programming.

With RAG, pre-trained, parametric-memory generation models are endowed with a non-parametric memory through a general-purpose fine-tuning approach. This means that you can supply the most up to date content to a pre existing model and, without additional training, use this new material as context to adjust the answers of the pre-existing model to fit your needs.

## Ollama 

The main building block we will use is **Ollama**, which is a platform to interact with large language models.
Running the AI assistant needs the Ollama server to run the LLM model on which it is based: to install Ollama see these [instructions](https://ollama.com/download).

Consider the sequence of commands below: note that the first command below will kill all ollama processes already running on your system
```
pkill -f ollama
ollama serve &
ollama pull llama3.3:70b
```
The `ollama serve &` command will run Ollama in the background. If this command does not work because Ollama's default port (11434) is already in use, set `OLLAMA_HOST` appropriately, then run `ollama serve &` again. A way to set `OLLAMA_HOST` properly is to just increase by one the port number (so for example `export OLLAMA_HOST=127.0.0.1:11435`).  Then the `Llama3.3` model with 70 billion parameters is pulled.

To test that Ollama is working you can do:
```
ollama run llama3.3:70b
```
The above command will run the LLM  locally, you can interact with it through the prompt and then exit with `/bye`.


We will show two ways of creating the AI assistant depending on the number of users in your system.

### System with a limited number of users

In this case, we assume the system has a small number of users, and that it can sustain the case where all of them are running Ollama locally.
The bigger in terms of parameters the models pulled from Ollama, the larger the memory requirements, hence if the number of users is large and the models are big, you could quickly finish up all the memory in the system. This is why we recommend the approach below only if the amount of users on the system is limited.

#### Setting up

The first thing to do, is to install the necessary software requirements: one could do it using `conda` (see [here](https://github.com/amd/HPCTrainingDock/blob/ecb81e4d7055f8594d34743b59cdeb1923faf40b/extras/scripts/miniconda3_setup.sh#L166) for how we setup the `miniconda3` module invoked below):

```
module load miniconda3
conda create -y -n amd_ai_assistant
conda activate amd_ai_assistant
git clone https://github.com/amd/HPCTrainingExamples.git
cd HPCTrainingExamples/MLExamples/RAG_LangChainDemo
pip3 install -r requirements.txt 
```

The installation of the requirements (last line above) will take quite a bit of time. If you do not want to use `conda`, and feel like you are aware of what you keep in your `PYTHONPATH`, you can do this instead:

```
mkdir ai_assistant_deps
cd ai_assistant_deps
export AI_ASSISTANT_DEPS=$PWD
cd ..
git clone https://github.com/amd/HPCTrainingExamples.git
cd HPCTrainingExamples/MLExamples/RAG_LangChainDemo
pip3 install -r requirements.txt --target=$AI_ASSISTANT_DEPS
export PYTHONPATH=$AI_ASSISTANT_DEPS:$PYTHONPATH
```

Note that it is **very** important to specify the target option when doing the installation with `pip` because that's where you specify the installation directory. In this way you will (hopefully) avoid messing up other Python packages you may have already installed in your local directory. The last line above will add the packages you just installed to your `PYTHONPATH` so they will be visible when you want to do `import <package>` in your Python scripts.

In the present directory, there currently are three versions of the AI assistant script you can run:

1. amd_ai_assistant.py
2. instinct_chat.py
3. instinct_chat_4_llm.py

We recommend you begin with `amd_ai_assistant.py` since it is the most complete, optimized and up to date of the three scripts.

All the above scripts will implement a retrieval augmented generation (RAG) model by scraping the web to get information on AMD and AMD software to provide the necessary context to pre trained  LLMs to be able to answer AMD specific questions leveraging the info from those specific websites. We initially focus on `amd_ai_assistant.py` for the reasons mentioned above. To see what websites and content is scraped in `amd_ai_assistant.py`, look at the urls in the `scrape_all` function:

```
async def scrape_all(rocm_version):
    rocm_docs_url=rocm_version
    if rocm_version == "latest":
       rocm_docs_url=rocm_version
    else:
       rocm_docs_url=f"docs-{rocm_version}"
    main_urls = [
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
                              .
                              .
                              .
```
You can edit the above list adding or removing urls at your discretion. Note that you can decide the level of recursion by which links at the above urls will be scraped (default is one level of recursion):
```
TIMEOUT = 5  # seconds
MAX_DEPTH = 1
CRAWL_DELAY = 1  # seconds delay between requests to avoid overload
CONCURRENT_REQUESTS = 5  # Limit max concurrent requests for politeness
```
Above, if you modify `MAX_DEPTH` to two for example, starting from the above urls, the script will scrape links at those urls and then the links at the links.
Let's assume that Ollama is running effectively on the background and you pulled `LLama3.3:70b` (which is the LLM `mad_ai_assistant.py` will be relying on for RAG): to run the code do:
```
cd HPCTrainingExamples/MLExamples/RAG_LangChainDemo
python3 amd_ai_assistant.py --rocm-version <rocm_version> --scrape
```
The above flags will specify what ROCm version to pull the documentation of, and also that we want to force scraping: this is because the script will save the scraped data locally so that the next time you run the script it will not scrape again, unless you force it with the `--scrape` option. Without scraping again, you will be immediately be supplied the prompt to interact with the model, saving considerable time:
```
AMD AI Assistant Ready! Type your questions. Type 'exit', 'quit' or 'bye' to stop.

Prompt: 
```
The script called `instinct_chat.py` has either the option to be used from command line, or to use a web user interface. The default is to use the command line option, to use the web interface (provided by Gradio) run it with:
```
cd HPCTrainingExamples/MLExamples/RAG_LangChainDemo
python3 instinct_chat.py --webui
```
otherwise, just omit the `--webui` option and you will get the command line prompt. Then copy paste the link displayed on terminal to your browser and you will get to the web user interface. Note that if you are running this script on a cluster, you will need to take care of the proper ssh tunneling to be able to open the user interface from your local browser. The script `instinct_chat_4_llm.py` only works with the Gradio web interface so running it with:
```
cd HPCTrainingExamples/MLExamples/RAG_LangChainDemo
python3 instinct_chat_4_llm.py
``` 
will give you the web interface option by default. The above script considers `llama3.3:70b`, `gemma2:27b`, `mistral-large` and `phi3:14b` to provide four answers to your prompt that will be displayed side by side in the Gradio interface. Remember to pull all these four models with Ollama before running the script.

### System with a large number of users

On a system with a large number of users, having each one of them run Ollama locally might be prohibitive. In such a case it could be helpful to have Ollama run on a dedicated node and then have users hop onto a web interface to interact with the models. This can be done in various ways, here we report one way to achieve it: below we assume Ollama is already installed and Podman is used as containers manager:
1. Ssh to the host system (something similar to `ssh $USER@aac6.amd.com`)
2. Ssh to the compute node on the host system (this is where Ollama will run)
3. Add this line: `host: 0.0.0.0` to the `.ollama/config.yaml`
4. Run `export OLLAMA_HOST=0.0.0.0:<port_number>` (for instance the port number might be 11435)
5. Run `export OLLAMA_PORT=<port_number>`
6. Run: `ollama serve &` to have Ollama run in the background
7. Run: `ollama pull <some_model>`: this step is not striclty necessary as you will be able to pull models as admin user of the Open WebUI
8. Run: `podman pull ghcr.io/open-webui/open-webui:ollama`: this command will pull the image you will run
9. Run: `podman run -d   -p 3000:8080   -e OLLAMA_BASE_URL=http://<host_sys_IP_address>:<port_number>   --gpus all   -v open-webui:/app/backend/data   --name open-webui-ollama   --restart always   ghcr.io/open-webui/open-webui:ollama`: this command will run the container using the image pulled at the previous step
10. From your local machine run: `ssh -L 3000:<compute_node>:3000 <host address>` (for instance <host address> could be `aac6.amd.com`)
11. Type this in the address bar of your browser (such as Microsoft Edge): `localhost:3000`
12. Create an admin account and make sure to remember the password you set. This is all done locally so if you remove the Open WebUI data from your host system you will be allowed to start over (you will lose all the data though, so make sure to take note of the password).

#### Troubleshooting tips

If you encounter unexpected behavior while setting up Open WebUI here is something you can do:

1. Kill Ollama
```
ps aux | grep 'ollama serve'
sudo pkill -f "ollama serve"
```
2. Stop and remove the container on Podman
```
podman stop open-webui-ollama
podman rm open-webui-ollama
```
3. If you get `505:internal error` when accessing `localhost:3000`, keep refreshing the page and it should get you there

Careful to not remove the volume (that you can see by doing `podman volume ls`) otherwise you will lose all the local data such as knowledge base, admin login info, user list etc.
