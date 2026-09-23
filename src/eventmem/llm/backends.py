"""Model backends.

Hosted SDK calls block, so they run in a worker thread. Calling them inline
would freeze the whole runtime for every agent while one model call is in
flight, which would make every latency measurement a measurement of the slowest
agent's model.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Optional, Sequence


class _HostedLLM:
    def __init__(self, client: Any, deployment: str) -> None:
        self.client = client
        self.deployment = deployment
        self.calls = 0
        self.failures = 0
        self.prompt_chars = 0
        self.completion_chars = 0

    def _chat_sync(self, system: str, user: str, max_tokens: int,
                   temperature: float) -> str:
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
        self.calls += 1
        self.prompt_chars += len(system) + len(user)
        try:
            answer = await asyncio.to_thread(
                self._chat_sync, system, user, max_tokens, temperature
            )
            self.completion_chars += len(answer)
            return answer
        except Exception as exc:
            # A failed call must not kill the cascade: the error becomes the
            # agent's output so the chain continues and the failure is visible
            # in the transcript rather than as a silent gap.
            self.failures += 1
            return f"[llm error: {type(exc).__name__}: {exc}]"


class AzureLLM(_HostedLLM):
    name = "azure"

    def __init__(self, client: Any, deployment: str) -> None:
        super().__init__(client, deployment)
        self.name = f"azure({deployment})"


class OpenAILLM(_HostedLLM):
    name = "openai"

    def __init__(self, client: Any, model: str = "gpt-4o-mini") -> None:
        super().__init__(client, model)
        self.name = f"openai({model})"


class MockLLM:
    """Deterministic stand-in with no key and no network.

    Rule-based on the system prompt's role and the user text. Deterministic on
    purpose: a benchmark whose agent outputs vary run to run cannot attribute a
    difference in results to the thing being tested.

    ``latency_s`` simulates model time. It defaults to zero so benchmarks are
    fast; set it when the question is how the runtime behaves while agents are
    busy.
    """

    name = "mock"

    def __init__(self, latency_s: float = 0.0,
                 rules: Optional[Sequence] = None) -> None:
        self.latency_s = latency_s
        self.calls = 0
        self._rules = list(rules) if rules else list(_DEFAULT_RULES)

    async def chat(self, system: str, user: str, max_tokens: int = 300,
                   temperature: float = 0.3) -> str:
        self.calls += 1
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        role, text = system.lower(), user.lower()
        for role_keys, text_keys, answer in self._rules:
            if any(k in role for k in role_keys) and (
                not text_keys or any(k in text for k in text_keys)
            ):
                return answer
        return "Acknowledged."


#: (role keywords, text keywords, answer). Empty text keywords match anything.
_DEFAULT_RULES = (
    (("classifier", "sentiment"),
     ("angry", "furious", "terrible", "refund", "worst", "unacceptable"),
     "negative"),
    (("classifier", "sentiment"), ("thanks", "great", "love", "happy"), "positive"),
    (("classifier", "sentiment"), (), "neutral"),
    (("escalation",), (),
     "At risk. The customer is threatening to leave over a double charge and an "
     "unresolved refund."),
    (("triage", "priority"), (),
     "Priority high. Route to a human agent and acknowledge within the hour."),
    (("security", "review"), (),
     "Security risk: the query is built from raw user input, which allows SQL "
     "injection. Use parameterised queries."),
    (("draft", "reply", "respond"), (),
     "Thank you for reaching out. I am sorry for the trouble. A specialist will "
     "contact you shortly to resolve the refund."),
    (("research", "investigate"), (),
     "The schema needs tables for slots, patients and clinicians, with a foreign "
     "key on slot id."),
    (("plan",), (), "Plan: research the schema, implement it, then review."),
)


class EchoLLM:
    """Returns a deterministic hash-derived token per prompt.

    For load tests where the content is irrelevant but responses must be
    distinct and reproducible.
    """

    name = "echo"

    def __init__(self, latency_s: float = 0.0) -> None:
        self.latency_s = latency_s
        self.calls = 0

    async def chat(self, system: str, user: str, max_tokens: int = 300,
                   temperature: float = 0.3) -> str:
        self.calls += 1
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        return hashlib.md5(f"{system}|{user}".encode()).hexdigest()[:16]
