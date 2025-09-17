import os
from dotenv import load_dotenv
load_dotenv(override=True)

ENDPOINT = os.getenv("ENDPOINT_URL","http://localhost:8000/v1")
DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "Qwen/Qwen2.5-7B-Instruct")
SUBSCRIPTION_KEY = os.getenv("LLM_API_KEY")
API_VERSION = os.getenv("API_VERSION", None)

EVALUATION_MODEL = os.getenv("EVALUATION_MODEL", "Qwen/Qwen2.5-7B-Instruct")
EVALUATION_MODEL_ENDPOINT = os.getenv("EVALUATION_MODEL_ENDPOINT", "http://localhost:8000/v1/evaluate")
EVALUATION_MODEL_KEY = os.getenv("EVALUATION_MODEL_KEY", "a")