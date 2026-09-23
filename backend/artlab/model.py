"""
model.py — chooses which chat model the agent graph talks to, and what it costs.

There are two options, and the rest of the app can't tell them apart:

  * Real Claude (`claude-haiku-4-5`), through LangChain's `ChatAnthropic` wrapper. Costs money.
  * A fake model that replays canned text word by word. Costs nothing. The tests use it,
    and you can click through the UI with it by setting ARTLAB_FAKE_LLM=1 in .env.

Both are LangChain `BaseChatModel` objects with the same methods (`ainvoke`, `astream`, …).
That is the point of the fake: the graph, the API and the web page run exactly the same
code path either way, so a free test run is a real test of everything except Claude itself.

This file also owns the price table and the cost formula, so every dollar figure in the
app (trace panel footer, per-chat budget) comes from one place.
"""

import os
from itertools import cycle

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

# The cheapest current Claude model. Change it here and everything follows.
MODEL_ID = "claude-haiku-4-5"

# Anthropic list price in US dollars per 1,000,000 tokens: (input tokens, output tokens).
PRICE_PER_MTOK = (1.00, 5.00)

# What the fake model says, so it's obvious in the UI that no real call happened.
FAKE_REPLY = "This is the fake model talking. No API call was made, so this reply cost $0."


def make_model() -> BaseChatModel:
    """Return the model the app should use, based on the ARTLAB_FAKE_LLM setting.

    Called once, when the API server starts (see `api.create_app`).

      ARTLAB_FAKE_LLM=1   -> the free fake model
      anything else       -> real Claude; `ChatAnthropic` reads ANTHROPIC_API_KEY from the
                             environment, which api.py loads from the .env file
    """
    if os.getenv("ARTLAB_FAKE_LLM") == "1":
        return fake_model()

    # Imported here rather than at the top so fake mode and the tests never load the
    # Anthropic client at all.
    from langchain_anthropic import ChatAnthropic

    # max_tokens is a ceiling on how long one answer may be. It is not a cost: you only
    # pay for tokens actually generated.
    return ChatAnthropic(model=MODEL_ID, max_tokens=16000)


def fake_model(replies: list[str] = [FAKE_REPLY]) -> BaseChatModel:
    """Build a fake chat model that answers with `replies`, one per call, looping forever.

    `GenericFakeChatModel` takes an iterator of `AIMessage`s and hands out the next one on
    every call, streaming it word by word just like a real model streams tokens.
    `cycle(replies)` makes the list repeat, so the fake never runs out; the generator
    creates a fresh `AIMessage` for every call, so replies never share one object.

    Example: fake_model(["Hi", "Bye"]) answers "Hi", then "Bye", then "Hi" again, and so on.
    """
    return GenericFakeChatModel(messages=(AIMessage(r) for r in cycle(replies)))


def model_name(model: BaseChatModel) -> str:
    """Human-readable model name for the trace panel.

    `ChatAnthropic` stores its model ID in `.model`; the fake model has no such attribute,
    so it shows as "fake model".
    """
    return getattr(model, "model", "fake model")


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Convert token counts into US dollars using PRICE_PER_MTOK.

    Example: 1,200 input + 300 output tokens
             = (1,200 × $1 + 300 × $5) / 1,000,000 = $0.0027
    """
    price_in, price_out = PRICE_PER_MTOK
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
