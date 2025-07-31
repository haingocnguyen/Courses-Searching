import asyncio
import ollama

class LLMService:
    def __init__(self, model_name: str = "qwen3:4b"):
        """
        Service để gọi Ollama LLM.
        model_name: default model sẽ dùng nếu không truyền explicit vào generate().
        """
        self.model = model_name

    async def generate(self, prompt: str, model: str = None, **options) -> str:
        model_to_use = model or self.model
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: ollama.generate(
                model=model_to_use,
                prompt=prompt,
                options=options
            )
        )
        return result["response"]