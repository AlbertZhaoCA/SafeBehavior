import time
from transformers import AutoTokenizer, AutoModelForCausalLM,PreTrainedModel
from utils.inference import get_gpu
from utils.templates.model import vicuna_template
import torch
import gc

class LLM():
    def __init__(self, model: str | PreTrainedModel, system_prompt: str = "You are a helpful assistant", **kwargs):
        # if "platform" in kwargs:
        #     kwargs.pop("platform")
        # if "key" in kwargs:
        #     kwargs.pop("key")
        if isinstance(model, PreTrainedModel):
            model_name = model.name_or_path
            self.model = model
        else:
            model_name = model
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                **kwargs
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_name.startswith("lmsys"):
            self.tokenizer.chat_template = vicuna_template

        self.system_prompt = system_prompt
        self.messages = [
            {
                "role": "system",
                "content": system_prompt
            },
        ]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def __del__(self):
        """Delete the model from GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        try:
            if torch is not None:
                torch.cuda.empty_cache()
        except AttributeError:
            pass
        gc.collect()
        print("LLM instance deleted.")

    def chat(self, messages: str | list, raw: bool = False,n:int = 1):
        """Chat with the LLM (maintains the conversation history)"""
        if isinstance(messages, str):
            self.messages.append(
                {
                    "role": "user",
                    "content": messages
                }
            )
        elif isinstance(messages, list):
            for message in messages:
                self.messages.append(message)
        else:
            raise ValueError("Messages must be a string or a list of strings.")
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(device=self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            attention_mask=inputs.attention_mask,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=n
        )
        generated = outputs[:, inputs.input_ids.shape[-1]:]

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=not raw)
        output = decoded if n > 1 else decoded[0]
        self.messages.append(
            {
                "role": "assistant",
                "content": output
            }
        )

        return output

    def generate(self, prompt: str | list, raw: bool = False,n:int = 1):
        """Generate text from the LLM (do not maintain the conversation history)"""
        message = [{
                    "role": "system",
                    "content": self.system_prompt
            }]
        if not isinstance(prompt, list):
            message.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )
        else:
            for msg in prompt:
                message.append(msg)
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(device=self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            attention_mask=inputs.attention_mask,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=n
        )
        generated = outputs[:, inputs.input_ids.shape[-1]:]

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=not raw)

        return decoded if n > 1 else decoded[0]
    
    def batch_generate(self, prompts: list[str], raw: bool = False,n:int = 1):
        """Generate text from the LLM in batch (do not maintain the conversation history)"""
        system_prompt = {
                    "role": "system",
                    "content": self.system_prompt
            }
        messages = []
        for prompt in prompts:
            messages.append([system_prompt, {
                "role": "user",
                "content": prompt
            }])


        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted_prompt,padding=True,return_tensors="pt").to(device=self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            attention_mask=inputs.attention_mask,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,

            num_return_sequences=n
        )

        generated = outputs[:, inputs.input_ids.shape[-1]:]

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=not raw)

        return decoded

    def get_text_with_template(self, messages: list[dict[str, str]],add_special_tokens,add_generation_prompt):
        """Get the text with the template applied"""
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            add_special_tokens=add_special_tokens
        )
        return formatted_prompt