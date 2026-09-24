"""
agents/common.py — small helpers shared by more than one node.

`text_of` normalises a message's text; `ms_since` and `tokens_used` turn raw timing and usage data
into the numbers the trace panel shows. Nothing here is graph-specific: any node can import these
without pulling in the whole graph.
"""

import time

from langchain_core.messages import BaseMessage


def text_of(message: BaseMessage) -> str:
    """A message's text, whether `content` is a plain string or a list of content blocks
    (Claude's replies can be [{"type": "text", "text": "..."}, ...])."""
    if isinstance(message.content, str):
        return message.content
    return "".join(b.get("text", "") for b in message.content if isinstance(b, dict))


def ms_since(start: float) -> int:
    """Milliseconds elapsed since `start` (a `time.perf_counter()` reading). Used for trace timings."""
    return round((time.perf_counter() - start) * 1000)


def tokens_used(message: BaseMessage) -> tuple[int, int]:
    """(input tokens, output tokens) a model reply reports. Real Claude fills these in; the fake leaves them empty (0, 0)."""
    usage = getattr(message, "usage_metadata", None) or {}
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0)
