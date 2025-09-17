import aiohttp


class Ollama:
    def __init__(self, model, base_url="http://localhost:11434/api/generate"):
        self.model = model
        self.base_url = base_url

    async def generate(self, prompt, system_prompt=None):
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "system": system_prompt or "",
            "options": {"temperature": 0}
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "")
                else:
                    error_text = await response.text()
                    raise Exception(f"Error: {response.status} - {error_text}")

