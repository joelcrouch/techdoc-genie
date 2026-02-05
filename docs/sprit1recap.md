# Sprint 1 ADD a LLM

##  WHich one?
So my  sprint plan calls for using gpt-4,but i am still trying to avoid api costs during dev, so i will use phi-3 mini.  It should fit and run on the laptop gpu:
```
nvidia-smi
Tue Feb  3 10:51:30 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX 500 Ada Gener...    Off |   00000000:01:00.0 Off |                  N/A |
| N/A   55C    P0             14W /   35W |      14MiB /   4094MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            3628      G   /usr/lib/xorg/Xorg                        4MiB |
+-----------------------------------------------------------------------------------------+
```
so i run htis and check:
```
curl -fsSL https://ollama.com/install.sh | sh 
>>> Cleaning up old version at /usr/local/lib/ollama
[sudo] password for dell-linux-dev3: 
>>> Installing ollama to /usr/local
>>> Downloading ollama-linux-amd64.tar.zst
######################################################################## 100.0%
>>> Adding ollama user to render group...
>>> Adding ollama user to video group...
>>> Adding current user to ollama group...
>>> Creating ollama systemd service...
>>> Enabling and starting ollama service...
>>> NVIDIA GPU installed.
(techdoc-genie-venv) (base) dell-linux-dev3@dell-linux-dev3-Precision-3591:~/Projects/techdoc-genie$ ollama --version
ollama version is 0.15.4
 
and also(this one take a little while):
 ollama pull phi3:mini
pulling manifest 
pulling 633fc5be925f: 100% ▕████████████████████████████████████████████████████████████████████████████████████████▏ 2.2 GB                         
pulling fa8235e5b48f: 100% ▕████████████████████████████████████████████████████████████████████████████████████████▏ 1.1 KB                         
pulling 542b217f179c: 100% ▕████████████████████████████████████████████████████████████████████████████████████████▏  148 B                         
pulling 8dde1baf1db0: 100% ▕████████████████████████████████████████████████████████████████████████████████████████▏   78 B                         
pulling 23291dc44752: 100% ▕████████████████████████████████████████████████████████████████████████████████████████▏  483 B                         
verifying sha256 digest 
writing manifest 
success 
```
Cool.  I downloaded the small llm.  NOw we need to go thru the motions, and add a provider class for it.

So in src/agent/provider/ollama.py i add something like this.  Modeleed
```
from typing import Dict, Any, List
import requests
from requests.exceptions import ConnectionError

from .base import BaseLLMProvider
from ...utils.logger import setup_logger
from ...utils.config import get_settings

logger = setup_logger(__name__)

class OllamaProvider(BaseLLMProvider):
    """
    LLM provider for Ollama models.
    Assumes Ollama server is running locally.
    """
    def __init__(self, model_name: str = "phi3:mini", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.settings = get_settings() # Not strictly needed for Ollama, but keeps consistent with other providers
        
        # Verify Ollama server connectivity
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.settings.ollama_timeout)
            response.raise_for_status()
            logger.info(f"Successfully connected to Ollama server at {self.base_url}")
            
            available_models = [m['name'] for m in response.json()['models']]
            if self.model_name not in available_models:
                logger.warning(
                    f"Model '{self.model_name}' not found on Ollama server. "
                    f"Available models: {', '.join(available_models)}. "
                    f"Attempting to use anyway, but it might fail."
                )
        except ConnectionError:
            logger.error(
                f"Could not connect to Ollama server at {self.base_url}. "
                "Please ensure Ollama is running and the model is pulled."
            )
            raise
        except Exception as e:
            logger.error(f"Error connecting to Ollama server: {e}")
            raise

    def generate_text(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a response from the Ollama model.
        """
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.0),
                "num_ctx": kwargs.get("max_tokens", 2048) # Map max_tokens to num_ctx for context window
            }
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.settings.ollama_timeout)
            response.raise_for_status()
            
            result = response.json()
            return result['message']['content']
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API for model {self.model_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Ollama generation: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Ollama can also provide embeddings if configured, but for now
        this provider only focuses on text generation.
        """
        raise NotImplementedError("Embedding generation is not implemente

```