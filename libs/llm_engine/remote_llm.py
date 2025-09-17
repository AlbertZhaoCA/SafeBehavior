import asyncio
from tenacity import retry, stop_never, wait_exponential, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor
from .adapters import OpenAIAdapter, AzureOpenAIAdapter
from .types import Plateform, ResponseType
from config import ENDPOINT, DEPLOYMENT, SUBSCRIPTION_KEY

class LLM:
    def __init__(self, model: str, platform: Plateform = "VLLM", system_prompt: str = None, **kwargs):
        base_url = kwargs.get("base_url", None)
        key = kwargs.get("key", None)
        if base_url:
            global ENDPOINT
            ENDPOINT = base_url
        if key:
            global SUBSCRIPTION_KEY
            SUBSCRIPTION_KEY = key
        self.model = model
        self.adapter = self._init_adapter(platform)
        self.system_prompt = system_prompt
        self.messages = None
        

    def _init_adapter(self, plateform: Plateform):
        match plateform:
            case "Aliyun":
                base_url = ENDPOINT if ENDPOINT else "https://dashscope.aliyuncs.com/compatible-mode/v1"
                model = self.model if self.model else DEPLOYMENT 
                return OpenAIAdapter(model=model,base_url=base_url,key=SUBSCRIPTION_KEY)
            case "OpenAI":
                base_url = ENDPOINT if ENDPOINT else "https://api.openai.com/v1"
                model = self.model if self.model else DEPLOYMENT 
                return OpenAIAdapter(model=model,base_url=base_url,key=SUBSCRIPTION_KEY)
            case "VLLM":
                base_url = ENDPOINT if ENDPOINT else "http://localhost:8000/v1"
                model = self.model if self.model else DEPLOYMENT
                return OpenAIAdapter(model=model,base_url=base_url,key=SUBSCRIPTION_KEY)
            case "Azure":
                model = self.model if self.model else DEPLOYMENT
                return AzureOpenAIAdapter(model=model)
            case _:
                raise ValueError(f"Unsupported platform: {plateform}")

    async def run_batch_job(self, input_file_path: str, output_file_path: str, error_file_path: str = None):
        input_id = self.adapter.upload_file(input_file_path)
        batch_id = self.adapter.create_batch_job(input_id)

        while True:
            status = self.adapter.check_job_status(batch_id)
            if status == "completed":
                print("Batch job completed successfully.")
                break
            elif status == "failed":
                batch = self.adapter.client.batches.retrieve(batch_id=batch_id)
                print(f"Batch job failed with error: {batch.errors}")
                print("Batch job failed.")
                raise RuntimeError("Batch job failed.")
            await asyncio.sleep(5)

        self.adapter.download_results(self.adapter.get_output_id(batch_id), output_file_path)
        if error_file_path:
            error_id = self.adapter.get_error_id(batch_id)
            if error_id:
                self.adapter.download_errors(error_id, error_file_path)

    @retry(
        stop=stop_never,
        wait=wait_exponential(multiplier=2, min=10, max=100),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def batch_generate(self, prompts: list[str]) -> list[str]:
        """
        Generate responses in batch mode for a list of prompts.
        
        Args:
            prompts (list[str]): A list of input prompts.
        
        Returns:
            list[str]: A list of generated responses.
        """
        results = []
        loop = asyncio.get_event_loop()

        async def generate_batch():
            with ThreadPoolExecutor() as executor:
                tasks = [loop.run_in_executor(executor, self.generate, prompt) for prompt in prompts]
                return await asyncio.gather(*tasks)

        try:
            results = loop.run_until_complete(generate_batch())
        except Exception as e:
            print(f"Error during batch generation: {e}")
            raise

        return results

    @retry(
        stop=stop_never,
        wait=wait_exponential(multiplier=2, min=10, max=100),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def chat(self, messages: list[dict[str, str]]|str, **kwargs) -> str:
        """generate response (maintain history)
        Args:
            messages(list[dict[str, str]]|str):message, follow system, user, assistant apprroach
            **kwargs: other arguments you want to pass to the adapter
        """
        if self.messages is None and self.system_prompt:
            self.messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                }
            ]
        elif self.messages is None:
            self.messages = []

        try:
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
            res = self.adapter.generate(self.messages, **kwargs)
            if res:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": res
                    }
                )
        except Exception as e:
            print(f"Error generating response: {e}")
            raise         
        return res

    @retry(
        stop=stop_never,
        wait=wait_exponential(multiplier=2, min=10, max=100),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))     
    )
    def generate(self, messages: list[dict[str, str]]|str,type:ResponseType=None,**kwargs) -> str:
        """generate response (do not maintain history)
        Args:
            messages(list[dict[str, str]]|str):message, follow system, user, assistant apprroach
            type('text' | 'json_object'): the type of the response
            **kwargs: other arguments you want to pass to the adapter
        """
        try:
            if isinstance(messages, str):
                if self.system_prompt:
                    messages = [
                        {
                            "role": "system",
                            "content": self.system_prompt
                        },
                        {
                            "role": "user",
                            "content": messages
                        }
                    ]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": messages
                        }
                    ]
            elif isinstance(messages, list):
                if self.system_prompt:
                    messages = [
                        {
                            "role": "system",
                            "content": self.system_prompt
                        }
                    ] + messages
                else:
                    messages = messages
            res = self.adapter.generate(messages, type=type, **kwargs)
        except Exception as e:
            print(f"Error generating response: {e}")
            raise         
        return res
    
    async def async_generate(self, messages: list[dict[str, str]] | str, type: ResponseType = None, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        
        result = None
        exception = None

        def _run_generate():
            nonlocal result, exception
            try:
                result = self.generate(messages, type, **kwargs)
            except Exception as e:
                exception = e

        try:
            future = loop.run_in_executor(None, _run_generate)
            await future
        except asyncio.CancelledError:
            print("Asynchronous generation task was cancelled.")
            raise
        except Exception as e:
            print(f"Error generating response asynchronously: {e}")
            raise
        
        if exception is not None:
            raise exception
            
        return result