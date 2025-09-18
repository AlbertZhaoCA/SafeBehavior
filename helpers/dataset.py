from utils.data import get_dataset
from datasets import load_dataset
from utils.templates.attack import build_bypass_attack_prompt

def get_attack_prompts(dataset:str,split:str,attacker:str):
    """
    Helper to get the attack prompts from the dataset.
    Args:
        dataset (str): The dataset to get the attack prompts from.
        split (str): The split of the dataset to use.
        attacker (str): The attacker to use.
    """
    if attacker == "DeepInception":
        attack_dataset = get_dataset(dataset, type="csv", split="train")
        attack_prompts = []
        for prompt in attack_dataset:
            if isinstance(prompt, dict):
                prompt = [prompt['prompt']]
            attack_prompts.append(prompt)
    elif attacker =="gcg":
        attack_dataset = get_dataset(dataset, type="csv", split="train") # supposed we load it from local
        attack_prompts = []
        for prompt in attack_dataset:
            if isinstance(prompt, dict):
                prompt = [prompt['prompt']]
            attack_prompts.append(prompt)
    elif attacker == "ifsj":
        attack_dataset = get_dataset(dataset, type="csv", split="train")
        attack_prompts = []
        for prompt in attack_dataset:
            if isinstance(prompt, dict):
                prompt = [prompt['prompt']]
            attack_prompts.append(prompt)
    elif attacker == "pap":
        attack_dataset = get_dataset(dataset, type="csv", split="train")
        attack_prompts = []
        for prompt in attack_dataset:
            if isinstance(prompt, dict):
                prompt = [prompt['prompt']]
            attack_prompts.append(prompt)
    elif attacker == "benign":
        attack_dataset = get_dataset(dataset, type="csv", split="train")
        attack_prompts = []
        for prompt in attack_dataset:
            if isinstance(prompt, dict):
                prompt = [prompt['Goal']]
            attack_prompts.append(prompt)
    elif attacker == "bypass":
        attack_prompts = []
        attack_dataset = get_dataset(dataset, type="csv", split="train")
        for prompt in attack_dataset:
            if isinstance(prompt, dict):
                goal = prompt['goal']
                attack_prompt = build_bypass_attack_prompt(goal)
                attack_prompts.append([attack_prompt])
        
    return attack_prompts

if __name__ == "__main__":
    """for testing purposes"""
    deepinception_prompts = get_attack_prompts("UWNSL/SafeDecoding-Attackers", "train", "DeepInception")
    print("DeepInception prompts:")
    for prompt in deepinception_prompts:
        print(prompt)
    gcg_prompts = get_attack_prompts("data/gcg/Qwen_Qwen2.5-7B-Instruct.csv", split="train",attacker="gcg")
    print("gcg prompts:")
    for prompt in gcg_prompts:
        print(prompt)
    print("\n")
  
