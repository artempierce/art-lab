"""
tools/agents.py — turns one worker into a tool another worker can call (Phase 9b, docs/contracts.md
§ 13, ticket H2). "Agent calls agent": content_ideator can hand youtube_researcher a question mid-turn
and read back its answer, the same way it reads back any other tool's result.

`make_agent_tool(model, tools, callee, prompt)` builds `ask_<callee>`, an **async** tool that runs a
whole nested `agents/tool_loop.py` turn for `callee` — its own system prompt, its own allowed tools,
its own MAX_TOOL_CALLS budget — and hands the answer back as an `AgentAnswer`.

Why the tool function itself has to be `async def`, not a plain one: it doesn't just compute something
and return — it *awaits* `run_tool_loop`, which awaits the model and, for each tool `callee` calls,
awaits `tools.call` again. A plain function can't contain an `await` at all, and running it in a worker
thread (tools/registry.py's usual path for a blocking tool) would drop it out of the async context
`get_stream_writer()` needs to write trace lines. So the tool gateway awaits a coroutine function like
this one directly, in the same async context — see tools/registry.py's `_run` for that half of it.

Depth and the shared cost budget are the *loop's* job, not this file's: `run_tool_loop(..., depth=1)`
is what refuses a nested `ask_*` request and what folds the nested run's `spent_usd` into the caller's
own total (agents/tool_loop.py). This file only has to pass the caller's current taint status through —
which it reads from a contextvar `run_tool_loop` sets for exactly the span of this call (see
`agents/tool_loop.py`'s module docstring for why a contextvar, not a new tool argument).
"""

from dataclasses import dataclass
from typing import Awaitable, Callable

from langchain_core.language_models import BaseChatModel

from artlab.tools.registry import ToolRegistry
from artlab.tools.untrusted import Piece


@dataclass(frozen=True)
class AgentAnswer:
    """What asking another agent a question gives back (docs/contracts.md § 13).

    text       the callee's final answer — plain text, no tool calls left unresolved — what the
               caller's own model gets to read, once the gateway wraps it (see `.pieces()` below)
    spent_usd  every model call the nested loop made, added into the caller's own `LoopResult.spent_usd`
               (agents/tool_loop.py) — the "shared budget": one turn's total cost, wherever it was spent
    tainted    whether the callee's own turn read any untrusted content. Kept here for completeness —
               what actually taints the *caller's* chat is this tool's own `untrusted_output=True`
               registration (tools/catalog.py): the researcher's answer is itself untrusted output,
               whether or not the research that produced it happened to be
    callee     which agent answered, e.g. "youtube_researcher" — used only by `.pieces()`, below

    Example: asking youtube_researcher "what's trending in desk setups?" might give back
             AgentAnswer(text="Desk setups are trending...", spent_usd=0.002, tainted=True,
             callee="youtube_researcher").
    """

    text: str
    spent_usd: float
    tainted: bool
    callee: str

    def pieces(self) -> list[Piece]:
        """One piece, labelled `agent:<callee>` (e.g. "agent:youtube_researcher") so the gateway's
        wrapping (tools/registry.py) says which agent this text came from — the same job a tool's own
        name plays for an ordinary `Piece`, e.g. "query_youtube_trends"."""
        return [Piece(source=f"agent:{self.callee}", text=self.text)]


def make_agent_tool(
    model: BaseChatModel, tools: ToolRegistry, callee: str, prompt: str
) -> Callable[[str], Awaitable[AgentAnswer]]:
    """Build `ask_<callee>(question: str) -> AgentAnswer`, ready to register with `tools.register(...)`
    (docs/contracts.md § 13, tools/catalog.py registers `ask_youtube_researcher` this way).

    Args:
        model:  the same chat model every worker's own tool loop uses — there's only one model in the
                app (model.py), so the nested loop `callee` runs on is no different from its own turn
        tools:  the shared tool gateway — `callee` gets exactly the tools its own allow-list grants,
                the same registry the caller's own loop is running against
        callee: the agent to ask, e.g. "youtube_researcher" — also `AgentAnswer.callee`, and what
                names the trace stage every line of the nested loop is written under
        prompt: `callee`'s own system prompt (its module's PROMPT constant) — the nested loop is a
                real turn for `callee`, not a shortcut, so it needs the same instructions `callee`'s
                own worker node would give it

    A late import, not a top-level one: `agents/tool_loop.py` sits on the *agents* side of the app, and
    `tools/catalog.py` — which calls this factory while building the *tools* side — is read by `api.py`
    before any agent module needs anything back from `tools/`. There's no cycle either way today (this
    module needs nothing else from `agents/tool_loop.py` at import time, only inside `ask`, once a tool
    is actually being built), but importing it here rather than at the top keeps that true for free —
    `callee`'s own module (e.g. `agents/youtube_researcher.py`, for its PROMPT) is a plain top-level
    import in `tools/catalog.py`, for the same reason: nothing there depends on `tools/` either.
    """
    from artlab.agents.tool_loop import caller_ctx, run_tool_loop, tainted_in_ctx

    async def ask(question: str) -> AgentAnswer:
        """The tool itself: hand `question` to `callee` as a brand-new, standalone turn, and return
        its answer. `depth=1` (agents/tool_loop.py, docs/contracts.md § 13) means `callee` gets every
        tool its allow-list grants *except* another `ask_*` — the depth cap that keeps agents from
        calling agents more than one level deep.

        `tainted_in` and `caller` come from contextvars the *caller's* own `run_tool_loop` set right
        before awaiting this call (see that module's docstring): this chat's current taint status, and
        which agent is doing the asking — the latter only for the nested loop's own trace lines
        ("↳ for content_ideator · …", docs/contracts.md § 13).
        """
        result = await run_tool_loop(
            model, tools, callee, prompt, question,
            tainted_in=tainted_in_ctx.get(), depth=1, caller=caller_ctx.get(),
        )
        return AgentAnswer(
            text=str(result.reply.content), spent_usd=result.spent_usd, tainted=result.tainted, callee=callee,
        )

    ask.__name__ = f"ask_{callee}"
    return ask
