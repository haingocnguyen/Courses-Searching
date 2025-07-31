import ollama
import json
from functools import lru_cache

class LLMAssistant:
    @lru_cache(maxsize=1000)
    def generate(self, prompt: str, model: str = "qwen2.5:1.5b") -> dict:
        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                format="json",
                options={"temperature": 0.4}
            )
            return json.loads(response['response'])
        except Exception as e:
            print(f"LLM Error: {str(e)}")
            return {}