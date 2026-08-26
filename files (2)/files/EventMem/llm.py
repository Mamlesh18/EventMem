"""
llm.py

The reasoning behind an agent. AzureLLM wraps your Azure OpenAI client. The
client call is blocking, so we run it in a worker thread with asyncio.to_thread.
That matters here: if we called it directly, the whole runtime would freeze
during every model call and no pushes could be delivered. Running it off the
event loop keeps the delivery machinery responsive while an agent thinks.

MockLLM lets the demo run with no key at all. It returns short canned answers so
you can watch the push and cascade work before spending a single token.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional, Protocol


class LLM(Protocol):
    async def chat(self, system: str, user: str, max_tokens: int = 300,
                   temperature: float = 0.3) -> str:
        ...


class AzureLLM:
    def __init__(self, client, deployment: str) -> None:
        self.client = client
        self.deployment = deployment

    def _chat_sync(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    async def chat(self, system: str, user: str, max_tokens: int = 300,
                   temperature: float = 0.3) -> str:
        try:
            return await asyncio.to_thread(
                self._chat_sync, system, user, max_tokens, temperature
            )
        except Exception as exc:  # keep the cascade alive if one call fails
            return f"[llm error: {exc}]"


class MockLLM:
    """
    A stand in that reads the role from the system prompt and returns a short,
    plausible answer. Good enough to prove the plumbing without a key.
    """

    async def chat(self, system: str, user: str, max_tokens: int = 300,
                   temperature: float = 0.3) -> str:
        await asyncio.sleep(0.01)  # imitate a little network latency
        low = system.lower()
        if "classifier" in low:
            text = user.lower()
            if any(w in text for w in ("angry", "furious", "terrible", "refund", "worst", "unacceptable")):
                return "negative"
            if any(w in text for w in ("thanks", "great", "love", "happy")):
                return "positive"
            return "neutral"
        if "escalation" in low:
            return "Yes, at risk. The customer is threatening to leave over a double charge and a refund that has not happened."
        if "triage" in low or "priority" in low:
            return "Priority high. Route to a human agent and acknowledge the complaint within the hour."
        if "security" in low or "review" in low:
            return "Security risk. The query is built from raw user input, which allows sql injection. Use parameters."
        if "draft" in low or "reply" in low or "respond" in low:
            return "Thank you for reaching out. I am sorry for the trouble. A specialist will contact you shortly to resolve the refund."
        return "Acknowledged."
