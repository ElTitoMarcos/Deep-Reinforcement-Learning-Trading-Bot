from __future__ import annotations

import time
import logging
from typing import Any

from ..utils.credentials import load_openai_key


class LLMClient:
    """Minimal client for calling a hosted LLM service.

    The implementation currently targets the OpenAI chat completions API but
    keeps the interface generic so that future providers could be supported.
    """

    def __init__(self, provider: str = "openai", model: str = "gpt-4o"):
        self.provider = provider
        self.model = model
        self.api_key = None
        self._client: Any | None = None
        self.logger = logging.getLogger(__name__)

        if provider == "openai":  # lazily import to avoid hard dependency
            try:  # pragma: no cover - optional dependency
                import openai

                self._client = openai
                try:
                    self.api_key = load_openai_key()
                    openai.api_key = self.api_key
                except Exception:
                    self.api_key = None
            except Exception:  # pragma: no cover - optional dependency
                self._client = None

    # ------------------------------------------------------------------
    def choose_model(self, model: str) -> None:
        """Update the active model identifier."""

        self.model = model

    # ------------------------------------------------------------------
    def ask(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1500,
    ) -> str:
        """Send a chat completion request and return the text response."""

        if self.provider != "openai":  # pragma: no cover - future providers
            raise RuntimeError(f"Unsupported provider: {self.provider}")

        if self._client is None:
            raise RuntimeError("openai package not installed or API key missing")

        backoff = 1.0
        for _ in range(5):
            try:
                response = self._client.ChatCompletion.create(  # type: ignore[attr-defined]
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                break
            except Exception as e:  # pragma: no cover - network/HTTP issues
                if getattr(e, "status", None) == 429 or "timeout" in str(e).lower():
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise
        else:  # pragma: no cover - all retries failed
            raise RuntimeError("LLM request failed after retries")

        text = response.choices[0].message["content"]  # type: ignore[index]

        # log cost if usage is available
        usage = getattr(response, "usage", None)
        if usage:
            cost = self._estimate_cost(self.model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            self.logger.info("llm_usage", model=self.model, **usage, cost_usd=cost)
        else:
            self.logger.info("llm_usage", model=self.model)

        return text

    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Rudimentary cost estimation in USD.

        Rates are based on public information and may become outdated.
        """

        pricing = {
            "gpt-4o": (5e-6, 15e-6),
            "gpt-4o-mini": (1e-6, 2e-6),
            "gpt-4.1": (5e-6, 15e-6),
            "gpt-4.1-mini": (1e-6, 3e-6),
        }
        in_rate, out_rate = pricing.get(model, (0.0, 0.0))
        return prompt_tokens * in_rate + completion_tokens * out_rate
