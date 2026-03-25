"""
Agent tools — pure data tools that do NOT touch the sandbox.

Build agent tools:  create_app, web_search
Edit agent tools:   modify_app, chat_message, web_search
"""

import os
from langchain_core.tools import tool


def _normalize_file_content(content: str) -> str:
    """Normalize escaped newlines in file content."""
    if not isinstance(content, str):
        return ""
    if "\\n" in content:
        return content.replace("\\n", "\n").replace('\\"', '"')
    return content


# ── Build agent tool ──────────────────────────────────────

@tool
def create_app(files: list[dict]) -> dict:
    """Generate project files. ALWAYS call this tool to output code.

    Args:
        files: List of {path: str, content: str} dicts.
    """
    if not isinstance(files, list):
        return {"files": []}

    normalized = [
        {"path": f["path"], "content": _normalize_file_content(f["content"])}
        for f in files
        if isinstance(f, dict) and "path" in f and "content" in f
    ]

    print(f"Tool: create_app — Generated {len(normalized)} files")
    return {"files": normalized}


# ── Edit agent tools ──────────────────────────────────────

@tool
def modify_app(files: list[dict]) -> dict:
    """Modify, create, or delete files in an existing project.

    Args:
        files: List of {path: str, content: str, action: "create"|"modify"|"delete"} dicts.
               Output COMPLETE file content for create/modify actions.
    """
    if not isinstance(files, list):
        return {"files": [], "success": False}

    normalized = [
        {
            "path": f["path"],
            "content": "" if f.get("action") == "delete" else _normalize_file_content(f.get("content", "")),
            "action": f.get("action", "modify"),
        }
        for f in files
        if isinstance(f, dict) and "path" in f
    ]

    print(f"Tool: modify_app — Processing {len(normalized)} file changes")
    return {"success": True, "files": normalized}


@tool
def chat_message(message: str) -> dict:
    """Send a chat message to the user.

    Args:
        message: The message to display to the user.
    """
    return {"type": "chat", "message": message}


# ── Web search (re-export) ───────────────────────────────

def get_web_search_tool():
    """Return web_search tool if SERPER_API_KEY is configured."""
    if os.getenv("SERPER_API_KEY"):
        from .web_search import create_web_search_tool
        return create_web_search_tool()
    return None


def get_build_tools() -> list:
    """Tools for the build agent: create_app + web_search."""
    tools = [create_app]
    ws = get_web_search_tool()
    if ws:
        tools.append(ws)
    return tools


def get_edit_tools() -> list:
    """Tools for the edit agent: modify_app + chat_message + web_search."""
    tools = [modify_app, chat_message]
    ws = get_web_search_tool()
    if ws:
        tools.append(ws)
    return tools
