from openai import OpenAI
import random
import threading
import time


class _RequestRateLimiter:
    """Space request starts across all worker threads and share 429 cooldowns."""

    def __init__(self, requests_per_minute: float | None):
        rpm = float(requests_per_minute or 0)
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self.next_request_at = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self.next_request_at - now)
            self.next_request_at = max(now, self.next_request_at) + self.min_interval
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def defer(self, seconds: float) -> None:
        """Pause future requests globally after the provider reports a limit."""
        with self.lock:
            self.next_request_at = max(
                self.next_request_at, time.monotonic() + max(0.0, seconds)
            )


class GPT:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        requests_per_minute: float | None = 30,
        max_attempts: int = 6,
        retry_base_seconds: float = 2,
    ):
        self.model_name = model
        # Disable SDK-internal retries so every attempt goes through the shared
        # limiter and the retry count remains truly bounded.
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        self.max_attempts = max(1, int(max_attempts))
        self.retry_base_seconds = max(0.1, float(retry_base_seconds))
        self.rate_limiter = _RequestRateLimiter(requests_per_minute)

    @staticmethod
    def _status_code(error: Exception) -> int | None:
        status_code = getattr(error, "status_code", None)
        if status_code is None:
            response = getattr(error, "response", None)
            status_code = getattr(response, "status_code", None)
        return status_code

    @staticmethod
    def _retry_after_seconds(error: Exception) -> float | None:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None
        value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _is_retryable(cls, error: Exception) -> bool:
        status_code = cls._status_code(error)
        if status_code in (408, 409, 429):
            return True
        if status_code is not None:
            return status_code >= 500
        return error.__class__.__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "RateLimitError",
            "TimeoutError",
        }

    def inference(self, prompt: str, temperature: float = 0.7) -> str:
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(1, self.max_attempts + 1):
            self.rate_limiter.wait()
            try:
                result = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                )
                content = result.choices[0].message.content
                if content is None:
                    raise ValueError("LLM returned an empty response")
                return content
            except Exception as e:
                if attempt >= self.max_attempts or not self._is_retryable(e):
                    print(
                        f"API call failed after {attempt} attempt(s); "
                        f"not retrying: {e}"
                    )
                    raise

                retry_after = self._retry_after_seconds(e)
                if retry_after is not None:
                    delay = retry_after + random.uniform(0, 1)
                else:
                    delay = min(
                        60.0, self.retry_base_seconds * (2 ** (attempt - 1))
                    )
                    delay *= random.uniform(0.8, 1.2)
                if self._status_code(e) == 429:
                    self.rate_limiter.defer(delay)
                print(
                    f"API call failed ({attempt}/{self.max_attempts}); "
                    f"retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)
