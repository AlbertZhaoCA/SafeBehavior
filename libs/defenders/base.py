from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM
from libs.llm_engine.llm import LLM

class BaseDefender(ABC):
    """
    Base class for all defenders.
    """
    def __init__(self, model: str | AutoModelForCausalLM = None):
        """
        Initialize the defender with a model.
        Args:
            llm : LLM model to be used for the defender.
        """
        self.llm : LLM = None

    @abstractmethod
    def run(self, prompts: str | list, **kwargs):
        """
        Defend the inputs using the defender's method.
        Args:
            prompts (str | list): The input prompts to defend.
            **kwargs: Additional arguments for the defender's method.
        """
        pass
