"""Which chat model the graph talks to: real Claude, or a free fake one for UI work and tests."""

import os
from itertools import cycle

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

MODEL_ID = "claude-haiku-4-5"
PRICE_PER_MTOK = (1.00, 5.00)  # USD per million tokens: (input, output)

FAKE_REPLY = "This is the fake model talking. No API call was made, so this reply cost $0."


def make_model() -> BaseChatModel:
    if os.getenv("ARTLAB_FAKE_LLM") == "1":
        return fake_model()
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=MODEL_ID, max_tokens=16000)


def fake_model(replies: list[str] = [FAKE_REPLY]) -> BaseChatModel:
    """Streams the given replies word by word, in a loop. Never calls an API."""
    return GenericFakeChatModel(messages=(AIMessage(r) for r in cycle(replies)))


def model_name(model: BaseChatModel) -> str:
    return getattr(model, "model", "fake model")


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICE_PER_MTOK
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
