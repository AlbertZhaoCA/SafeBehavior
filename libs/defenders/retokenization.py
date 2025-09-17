from utils.bpe import BpeOnlineTokenizer,load_subword_nmt_table
from .base import BaseDefender
from ..llm_engine.llm import LLM
from .registry import register_defender

@register_defender("retokenization")
class RetokenizationDenfender(BaseDefender):
    merge_table = None

    def __init__(self,model,type="local",bpe_path="data/subword_nmt.voc",dropout_rate=0.2, **kwargs):
        if RetokenizationDenfender.merge_table is None:
            RetokenizationDenfender.merge_table = load_subword_nmt_table(bpe_path)
        self.llm = LLM(model=model,type=type,**kwargs)
        self. subword_nmt_tokenizer = BpeOnlineTokenizer(
            bpe_dropout_rate = dropout_rate,
            merge_table = RetokenizationDenfender.merge_table)
        
    def run(self,prompts: str):
        """
        Defend the inputs using the self exam method.
        Args:
            prompts (str): The input prompts to defend.
        """
        user_prompt_retokenized = self.subword_nmt_tokenizer(prompts, 
                sentinels=['', '</w>'],
                regime='end',
                bpe_symbol=' ')
        print(user_prompt_retokenized)
        output = self.llm.generate(user_prompt_retokenized)
        return output
    
if __name__ == "__main__":
    from utils.data import get_dataset
    """for testing purposes"""
    self_remainder = RetokenizationDenfender(model="Qwen/Qwen2.5-7B")
    attack_prompts = get_dataset('flydust/SafeDecoding-Attackers',split='train',source="DeepInception",)
    for prompt in attack_prompts:
        prompt = prompt['prompt']
        print(prompt)
        output = self_remainder.run(prompt)
        print(output)