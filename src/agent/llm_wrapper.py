from typing import Any, List, Mapping, Optional

from langchain_core.language_models import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun

from .providers.base import BaseLLMProvider
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class CustomLLM(LLM):
    """
    Custom LLM class to wrap our BaseLLMProvider for LangChain compatibility.
    """
    llm_provider: BaseLLMProvider
    model_name: str

    @property
    def _llm_type(self) -> str:
        return self.model_name

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Calls the underlying BaseLLMProvider's generate_text method.
        """
        if stop is not None:
            logger.warning("`stop` keyword argument is not implemented for CustomLLM.")
        
        response = self.llm_provider.generate_text(prompt, **kwargs)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Return the identifying parameters.
        """
        return {"model_name": self.model_name, "provider_type": self.llm_provider.__class__.__name__}
