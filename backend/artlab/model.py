"""
model.py — chooses which chat model the agents talk to, and what it costs.

There are two options, and the rest of the app can't tell them apart:

  * Real Claude (`claude-haiku-4-5`), through LangChain's `ChatAnthropic` wrapper. Costs money.
  * `FakeChatModel`, a free stand-in with predictable behaviour. The tests use it, and you can click
    through the whole UI with it by setting ARTLAB_FAKE_LLM=1 in .env.

Both are LangChain chat models with the same methods (`ainvoke`, `with_structured_output`, …), so
the graph, the API and the web page run exactly the same code path either way. With the fake, the
knowledge-base search is still completely real (local embeddings) — only the *writing* is faked.

This file also owns the price table and the cost formula, so every dollar figure in the app
(trace panel footer, per-chat budget) comes from one place.
"""

import itertools
import json
import os
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, PrivateAttr

# The cheapest current Claude model. Change it here and everything follows.
MODEL_ID = "claude-haiku-4-5"

# Anthropic list price in US dollars per 1,000,000 tokens: (input tokens, output tokens).
PRICE_PER_MTOK = (1.00, 5.00)

# What the fake model says, so it's obvious in the UI that no real call happened.
FAKE_REPLY = "This is the fake model talking. No API call was made, so this reply cost $0."

# Words that make the fake supervisor send a question to rag_agent (a crude stand-in for Claude's judgement).
KNOWLEDGE_HINTS = re.compile(
    r"\b(our|policy|policies|rules?|guidelines?|requirements?|checklist|process|sponsor\w*|expenses?|"
    r"moderation|upload\w*|calendar|approv\w*|brand|studio|architecture|personas?)\b",
    re.IGNORECASE,
)


def make_model() -> BaseChatModel:
    """Return the model the app should use, based on the ARTLAB_FAKE_LLM setting.

    Called once, when the API server starts (see `api.create_app`).

      ARTLAB_FAKE_LLM=1   -> the free fake model
      anything else       -> real Claude; `ChatAnthropic` reads ANTHROPIC_API_KEY from the
                             environment, which api.py loads from the .env file
    """
    if os.getenv("ARTLAB_FAKE_LLM") == "1":
        return fake_model()

    # Imported here rather than at the top so fake mode and the tests never load the Anthropic
    # client at all.
    from langchain_anthropic import ChatAnthropic

    # max_tokens is a ceiling on how long one answer may be. It is not a cost: you only pay for
    # tokens actually generated.
    return ChatAnthropic(model=MODEL_ID, max_tokens=16000)


class FakeChatModel(BaseChatModel):
    """A free, predictable stand-in for Claude. It plays four roles:

      supervisor       when asked for a RouteDecision (structured output), it routes to rag_agent if
                       the question contains a KNOWLEDGE_HINTS word, otherwise to "respond"
      rag_agent        when its prompt contains <untrusted_retrieval> sources, it quotes the start of
                       source [1] and cites it — so you can see real retrieval results in the UI
      rag_agent retry  when asked for a Rephrase (structured output, after a first search found
                       nothing), it rewords the question by appending " policy" — deterministic, and
                       different enough from the original to plausibly match a second time
      respond          otherwise, it answers with the next of `replies`, in a loop

    How structured output works (and why `bind_tools` is here): LangChain's `with_structured_output(Schema)`
    turns the schema into a *tool* the model is forced to call, then reads the tool call's arguments
    back as a Schema object. Real Claude does that natively; this fake does it by remembering which
    tool it was bound to (`bind_tools`) and returning a tool call for it.
    """

    replies: list[str] = Field(default_factory=lambda: [FAKE_REPLY])
    tool_name: str | None = None  # set by bind_tools: the structured-output schema we must "fill in"
    _turn: Any = PrivateAttr(default_factory=itertools.count)  # which reply comes next

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools: list, **kwargs: Any) -> "FakeChatModel":
        """Return a copy that answers by calling the first tool (what `with_structured_output` needs)."""
        tool = tools[0]
        name = tool.__name__ if isinstance(tool, type) else tool["name"]
        return self.model_copy(update={"tool_name": name})

    def _reply(self, messages: list[BaseMessage]) -> AIMessage:
        """Decide what to say, based on which role we're playing (see the class docstring)."""
        if self.tool_name == "RouteDecision":
            question = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
            to_rag = bool(KNOWLEDGE_HINTS.search(question))
            args = {
                "next": "rag_agent" if to_rag else "respond",
                "reason": "fake router: mentions the studio's docs" if to_rag else "fake router: general question",
                "question": question,
            }
            return AIMessage("", tool_calls=[{"name": self.tool_name, "args": args, "id": "fake-call", "type": "tool_call"}])
        if self.tool_name == "Rephrase":
            question = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
            args = {"query": f"{question} policy"}
            return AIMessage("", tool_calls=[{"name": self.tool_name, "args": args, "id": "fake-call", "type": "tool_call"}])
        if self.tool_name:
            raise ValueError(f"the fake model can't fill in {self.tool_name}")

        prompt = str(messages[-1].content)
        source = re.search(r"<untrusted_retrieval[^>]*>\n(.*?)\n</untrusted_retrieval>", prompt, re.DOTALL)
        if source:
            body = source.group(1).split("\n\n", 1)[-1]  # drop the "Title › Heading" line
            return AIMessage(f"(Fake model, no API call.) The knowledge base says: {body[:220].strip()} [1]")
        return AIMessage(self.replies[next(self._turn) % len(self.replies)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        """Answer in one piece (used when nothing is streaming)."""
        return ChatResult(generations=[ChatGeneration(message=self._reply(messages))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        """Answer word by word, like a real model streams tokens, so the UI shows text arriving."""
        reply = self._reply(messages)
        if reply.tool_calls:
            call = reply.tool_calls[0]
            yield ChatGenerationChunk(message=AIMessageChunk("", tool_call_chunks=[
                {"name": call["name"], "args": json.dumps(call["args"]), "id": call["id"], "index": 0}
            ]))
            return
        for piece in re.findall(r"\S+\s*|\s+", reply.content):  # words with their spacing, so they rejoin exactly
            chunk = ChatGenerationChunk(message=AIMessageChunk(piece))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk


def fake_model(replies: list[str] = [FAKE_REPLY]) -> FakeChatModel:
    """A FakeChatModel whose "respond" answers are `replies`, in a loop.

    Example: fake_model(["Hi", "Bye"]) answers "Hi", then "Bye", then "Hi" again, and so on.
    """
    return FakeChatModel(replies=list(replies))


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
