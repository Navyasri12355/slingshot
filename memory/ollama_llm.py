import requests


class OllamaLLM:
    def __init__(self, model="mistral"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str):

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False   # ⭐ VERY IMPORTANT
        }

        response = requests.post(self.url, json=payload)

        if response.status_code != 200:
            raise Exception(
                f"Ollama error: {response.text}"
            )

        data = response.json()

        return data.get("response", "")