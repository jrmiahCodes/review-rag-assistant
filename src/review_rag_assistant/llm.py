"""Swappable LLM backends behind one small interface.

A generator spec is "<provider>:<model>", e.g. "ollama:qwen3.5:4b" or
"anthropic:claude-sonnet-5-5". The rest of the code only sees `LLM`.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from review_rag_assistant.config import settings


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema for the arguments


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    name: str
    last_usage: Usage

    def stream(self, system: str, user: str) -> Iterator[str]: ...

    def tool_calls(self, system: str, user: str, tools: list[ToolSpec]) -> list[ToolCall]: ...


class OllamaLLM:
    def __init__(self, model: str, think: bool = False, num_ctx: int = 8192):
        import ollama

        self.client = ollama.Client()
        self.model = model
        self.name = f"ollama:{model}"
        # Thinking off by default: on a 4B model it mostly adds latency to a grounded summarization task.
        self.think = think
        self.options = {"num_ctx": num_ctx, "temperature": 0}
        self.last_usage = Usage()

    def stream(self, system: str, user: str) -> Iterator[str]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for part in self.client.chat(self.model, messages, stream=True, think=self.think, options=self.options):
            if part.message.content:
                yield part.message.content
            if part.done:
                self.last_usage = Usage(part.prompt_eval_count or 0, part.eval_count or 0)

    def tool_calls(self, system: str, user: str, tools: list[ToolSpec]) -> list[ToolCall]:
        res = self.client.chat(
            self.model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in tools
            ],
            think=self.think,
            options=self.options,
        )
        self.last_usage = Usage(res.prompt_eval_count or 0, res.eval_count or 0)
        return [ToolCall(c.function.name, dict(c.function.arguments)) for c in (res.message.tool_calls or [])]


class AnthropicLLM:
    def __init__(self, model: str, effort: str = "low"):
        import anthropic

        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = model
        self.name = f"anthropic:{model}"
        # Grounded Q&A over provided sources is closer to chat than to hard reasoning, so low effort.
        self.effort = effort
        self.last_usage = Usage()

    def stream(self, system: str, user: str) -> Iterator[str]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=4000,
            system=system,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": user}],
        ) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
        self.last_usage = Usage(final.usage.input_tokens, final.usage.output_tokens)
        if final.stop_reason == "refusal":
            yield "\n\n[The model declined to answer this request.]"

    def tool_calls(self, system: str, user: str, tools: list[ToolSpec]) -> list[ToolCall]:
        res = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system,
            output_config={"effort": self.effort},
            tools=[{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools],
            messages=[{"role": "user", "content": user}],
        )
        self.last_usage = Usage(res.usage.input_tokens, res.usage.output_tokens)
        return [ToolCall(b.name, dict(b.input)) for b in res.content if b.type == "tool_use"]


def get_llm(spec: str | None = None) -> LLM:
    spec = spec or settings.generator
    provider, _, model = spec.partition(":")
    if provider == "ollama":
        return OllamaLLM(model)
    if provider == "anthropic":
        return AnthropicLLM(model)
    raise ValueError(f"unknown generator provider in {spec!r} (expected ollama:... or anthropic:...)")
