"""
Build agent — LangGraph agent loop for initial app generation.

Ports Adorable's agent.ts to Python.
Graph: agent ⇄ tools, with conditional edge:
  AIMessage with no tool_calls → END
  AIMessage with tool_calls    → tools → agent
"""

import json
import re
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode

from .agent import create_build_llm
from .prompts import get_build_system_prompt
from .tools import get_build_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json_parse(value: str) -> Any:
    """Try to JSON-parse a string; return None on any failure."""
    try:
        return json.loads(value)
    except Exception:
        return None


def _build_graph(api_key: str):
    """Construct and compile the LangGraph agent graph for the build agent."""
    tools = get_build_tools()
    llm = create_build_llm(api_key).bind_tools(tools)
    tool_node = ToolNode(tools)

    async def agent_node(state: dict) -> dict:
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: dict) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and (not last.tool_calls or len(last.tool_calls) == 0):
            return END
        return "tools"

    graph = (
        StateGraph(MessagesState)
        .add_node("agent", agent_node)
        .add_node("tools", tool_node)
        .add_edge("__start__", "agent")
        .add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        .add_edge("tools", "agent")
        .compile()
    )
    return graph


def _messages_to_files(messages: list) -> list[dict]:
    """Scan ToolMessages for create_app results and merge files (last write wins by path)."""
    merged: dict[str, dict] = {}

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        content = msg.content
        if isinstance(content, str):
            parsed = _safe_json_parse(content)
            if parsed is not None:
                content = parsed

        if not content or not isinstance(content, dict):
            continue

        files = content.get("files")
        if not isinstance(files, list):
            continue

        for f in files:
            if isinstance(f, dict) and f.get("path") and f.get("content") is not None:
                merged[f["path"]] = {"path": f["path"], "content": f["content"]}

    return list(merged.values())


def _auto_wire_app_jsx(files: list[dict]) -> list[dict]:
    """
    If src/App.jsx is missing but at least one src/components/* file exists,
    create a minimal App.jsx that imports and renders the first component found.
    """
    paths = {f["path"] for f in files}
    if "src/App.jsx" in paths:
        return files

    component_file = next(
        (f for f in files if f["path"].startswith("src/components/")),
        None,
    )
    if not component_file:
        return files

    # Derive component name from filename (strip extension)
    component_name = component_file["path"].split("/")[-1]
    component_name = re.sub(r"\.\w+$", "", component_name)

    app_content = (
        f'import React from "react";\n'
        f'import {component_name} from "./components/{component_name}";\n'
        f'\n'
        f'export default function App() {{\n'
        f'  return (\n'
        f'    <div className="min-h-screen bg-background text-foreground">\n'
        f'      <{component_name} />\n'
        f'    </div>\n'
        f'  );\n'
        f'}}\n'
    )

    return files + [{"path": "src/App.jsx", "content": app_content}]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_build_stream(
    prompt: str,
    api_key: str,
    on_event: Callable[[dict], None],
) -> dict:
    """
    Run the build agent, streaming log events via on_event.

    Returns:
        {"success": True, "files": [...], "project_name": "..."}
        or
        {"success": False, "error": "..."}
    """
    system_prompt = get_build_system_prompt()
    initial_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]

    on_event({"e": "log", "message": "Building your application..."})

    graph = _build_graph(api_key)

    # Accumulate all messages so we can extract files after the stream ends.
    all_messages: list = list(initial_messages)

    try:
        async for chunk in graph.astream(
            {"messages": initial_messages},
            {"recursion_limit": 50},
        ):
            for node_name, state in chunk.items():
                new_msgs = state.get("messages", [])
                if not new_msgs:
                    continue

                if not isinstance(new_msgs, list):
                    new_msgs = [new_msgs]

                all_messages.extend(new_msgs)
                last_msg = new_msgs[-1]

                if node_name == "agent":
                    if isinstance(last_msg, AIMessage) and isinstance(last_msg.content, str) and last_msg.content:
                        on_event({"e": "log", "message": "Thinking..."})

                elif node_name == "tools":
                    # Find what tool was just called
                    last_ai = next(
                        (m for m in reversed(all_messages) if isinstance(m, AIMessage)),
                        None,
                    )
                    tool_name = (
                        last_ai.tool_calls[0]["name"]
                        if last_ai and last_ai.tool_calls
                        else "tool"
                    )

                    if tool_name == "web_search":
                        on_event({"e": "log", "message": "Searching the web..."})
                    elif tool_name == "create_app":
                        # Emit file count from the tool result
                        if isinstance(last_msg, ToolMessage):
                            content = last_msg.content
                            if isinstance(content, str):
                                content = _safe_json_parse(content)
                            if isinstance(content, dict) and isinstance(content.get("files"), list):
                                count = len(content["files"])
                                if count > 0:
                                    on_event({"e": "log", "message": f"Generated {count} files"})

    except Exception as exc:
        return {"success": False, "error": str(exc)}

    # Extract and merge files from all ToolMessages
    files = _messages_to_files(all_messages)

    if not files:
        last = all_messages[-1] if all_messages else None
        error_msg = (
            str(last.content)
            if isinstance(last, AIMessage)
            else "Agent failed to generate files."
        )
        return {"success": False, "error": error_msg}

    files = _auto_wire_app_jsx(files)

    # Derive project name from the first component file found
    component_file = next(
        (f for f in files if f["path"].startswith("src/components/")),
        None,
    )
    if component_file:
        raw_name = re.sub(r"\.\w+$", "", component_file["path"].split("/")[-1])
        # CamelCase → "Camel Case"
        project_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw_name)
    else:
        project_name = "Untitled Project"

    return {"success": True, "files": files, "project_name": project_name}
