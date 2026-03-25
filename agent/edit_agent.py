"""
Edit agent + error-fix agent — LangGraph (agent ⇄ tools) using o4-mini.

Tools: modify_app, chat_message, web_search
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode

from .agent import create_edit_llm
from .tools import get_edit_tools
from .prompts import get_edit_system_prompt, get_error_fix_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json_parse(value: str) -> Any | None:
    """Try to JSON-parse a string; return None on failure."""
    try:
        return json.loads(value)
    except Exception:
        return None


def _extract_file_changes(messages: list) -> list[dict]:
    """
    Walk all ToolMessages and collect file changes returned by modify_app.

    modify_app returns: {"success": True, "files": [{path, content, action}, ...]}
    """
    collected: list[dict] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        content = msg.content
        if isinstance(content, str):
            content = _safe_json_parse(content) or content

        if not isinstance(content, dict):
            continue

        files = content.get("files")
        if isinstance(files, list):
            for f in files:
                if isinstance(f, dict) and f.get("path"):
                    collected.append(f)

    return collected


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_edit_graph(api_key: str):
    """
    Build and compile a LangGraph agent for edit operations.

    Structure: agent → (conditional) → tools → agent → …
    """
    edit_llm = create_edit_llm(api_key)
    edit_tools = get_edit_tools()
    llm_with_tools = edit_llm.bind_tools(edit_tools)

    async def agent_node(state: dict) -> dict:
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: dict) -> str:
        messages = state.get("messages", [])
        if not messages:
            return END
        last = messages[-1]
        if isinstance(last, AIMessage) and not getattr(last, "tool_calls", None):
            return END
        return "tools"

    tool_node = ToolNode(edit_tools)

    workflow = (
        StateGraph(MessagesState)
        .add_node("agent", agent_node)
        .add_node("tools", tool_node)
        .add_edge("__start__", "agent")
        .add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        .add_edge("tools", "agent")
    )

    return workflow.compile()


# ---------------------------------------------------------------------------
# Streaming helpers (shared between edit and error-fix)
# ---------------------------------------------------------------------------

async def _stream_graph(
    graph,
    messages: list,
    on_event: Callable[[dict], None],
    file_event_key: str = "file_update",
) -> list[dict]:
    """
    Stream a compiled graph and emit SSE-style events via on_event.

    Returns the accumulated list of FileChange dicts.
    """
    collected_files: list[dict] = []
    seen_message_ids: set[str] = set()

    stream = graph.astream({"messages": messages}, {"recursion_limit": 50})

    async for chunk in stream:
        for _node_name, state in chunk.items():
            raw_messages = state.get("messages", [])
            if not isinstance(raw_messages, list):
                raw_messages = [raw_messages]

            for msg in raw_messages:
                # Deduplicate by id when available
                msg_id = getattr(msg, "id", None)
                if msg_id:
                    if msg_id in seen_message_ids:
                        continue
                    seen_message_ids.add(msg_id)

                # AIMessage with text content → token event
                if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content:
                    on_event({"e": "token", "content": msg.content})

                # ToolMessage → inspect result
                elif isinstance(msg, ToolMessage):
                    content = msg.content
                    if isinstance(content, str):
                        content = _safe_json_parse(content) if content else None

                    if isinstance(content, dict):
                        # modify_app result: {"success": True, "files": [...]}
                        files = content.get("files")
                        if isinstance(files, list):
                            for f in files:
                                if isinstance(f, dict) and f.get("path"):
                                    collected_files.append(f)
                                    on_event({
                                        "e": file_event_key,
                                        "file": {
                                            "path": f.get("path"),
                                            "action": f.get("action", "modify"),
                                        },
                                    })

                        # chat_message result: {"type": "chat", "message": "..."}
                        if content.get("type") == "chat" and content.get("message"):
                            on_event({"e": "token", "content": content["message"]})

    return collected_files


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_edit_stream(
    current_files: dict[str, str],
    user_message: str,
    chat_history: list[dict],
    api_key: str,
    on_event: Callable[[dict], None],
) -> list[dict]:
    """
    Run the edit agent with streaming, emitting events via on_event.

    Args:
        current_files:  {path: content} mapping of the current project.
        user_message:   The latest user request.
        chat_history:   Prior conversation turns [{role, content}, ...].
        api_key:        OpenRouter API key.
        on_event:       Callback for SSE events.

    Returns:
        List of FileChange dicts [{path, content, action}].
    """
    logger.info("run_edit_stream: starting")

    system_prompt = get_edit_system_prompt(current_files)

    messages: list = [
        SystemMessage(content=system_prompt),
        *[
            HumanMessage(content=turn["content"])
            if turn.get("role") == "user"
            else AIMessage(content=turn["content"])
            for turn in chat_history
        ],
        HumanMessage(content=user_message),
    ]

    graph = _build_edit_graph(api_key)

    try:
        file_changes = await _stream_graph(graph, messages, on_event, file_event_key="file_update")
        logger.info("run_edit_stream: finished — %d file change(s)", len(file_changes))
        return file_changes
    except Exception as exc:
        logger.exception("run_edit_stream error: %s", exc)
        on_event({"e": "error", "message": "Failed to process edit request"})
        return []


async def run_error_fix_stream(
    current_files: dict[str, str],
    build_errors: str,
    api_key: str,
    on_event: Callable[[dict], None],
) -> list[dict]:
    """
    Run the error-fix agent with streaming, emitting events via on_event.

    Args:
        current_files:  {path: content} mapping of the current project.
        build_errors:   Build / runtime error output to fix.
        api_key:        OpenRouter API key.
        on_event:       Callback for SSE events.

    Returns:
        List of FileChange dicts [{path, content, action}].
    """
    logger.info("run_error_fix_stream: starting")

    system_prompt = get_error_fix_prompt(current_files, build_errors)

    messages: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please fix the build errors shown above."),
    ]

    graph = _build_edit_graph(api_key)

    try:
        file_changes = await _stream_graph(graph, messages, on_event, file_event_key="file_update")
        logger.info("run_error_fix_stream: finished — %d file change(s)", len(file_changes))
        return file_changes
    except Exception as exc:
        logger.exception("run_error_fix_stream error: %s", exc)
        on_event({"e": "error", "message": "Failed to fix build errors"})
        return []
