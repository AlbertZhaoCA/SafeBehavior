from .base import BaseDefender
from ..llm_engine.llm import LLM
from .registry import register_defender

@register_defender("paraphrase")
class ParaphraseDefender(BaseDefender):
    def __init__(self, model,type="local",**kwargs):
        self.llm = LLM(model=model,type=type,**kwargs)
    

    def run(self, prompts):
        """
        Defend the inputs using the paraphrase method.
        Args:
            prompts (str | list): The input prompts to defend.
        """
        input_prompt_paraphrase = f"Please paraphrase the following sentences. Give me paraphrased results only. Do not include any other information.\n{prompts}" 
        output = self.llm.generate(input_prompt_paraphrase)
        output = self.llm.generate(output)
        return output

if __name__ == "__main__":
    from utils.data import get_dataset
    """for testing purposes"""
    self_remainder = ParaphraseDefender(model="Qwen/Qwen2.5-7B")
    attack_prompts = get_dataset('flydust/SafeDecoding-Attackers',split='train',source="DeepInception",)
    for prompt in attack_prompts:
        prompt = prompt['prompt']
        print(prompt)
        output = self_remainder.run(prompt)
        print(output)