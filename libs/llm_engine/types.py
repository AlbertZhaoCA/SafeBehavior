from typing import Literal

Plateform = Literal["Aliyun", "Azure", "OpenAI", "Google", "AWS", "HuggingFace"]

ModelType = Literal["local", "remote","ollama"]

ResponseType = Literal["text", "json_object"]