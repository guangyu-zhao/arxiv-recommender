from openai import OpenAI
import time


class GPT:
    def __init__(self, model: str, base_url: str, api_key: str):
        self.model_name = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def inference(self, prompt: str, temperature: float = 0.7) -> str:
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(10):
            try:
                result = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                )
                return result.choices[0].message.content
            except Exception as e:
                if attempt < 9:
                    print(f"API call failed ({attempt + 1}/10), retrying in 1s: {e}")
                    time.sleep(1)
                else:
                    print(f"API call failed after 10 attempts: {e}")
                    raise