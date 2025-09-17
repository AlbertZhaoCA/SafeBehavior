from .base import BaseDefender
from ..llm_engine.llm import LLM
from .registry import register_defender

@register_defender("vanilla")
class VanillaDefender(BaseDefender):
    def __init__(self,model,type="local",system_prompt=None,**kwargs):
        self.llm = LLM(model=model,type=type,system_prompt=system_prompt,**kwargs)
    
    def run(self,prompts: str) -> str:
        """
        Defend the inputs using the vanilla method, which means no defense.
        Args:
            prompts (str): The input prompts to defend.
        """
        output = self.llm.generate(prompts)
        return output
    