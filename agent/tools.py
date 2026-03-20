from e2b_code_interpreter import AsyncSandbox
import asyncio
from langchain_core.tools import tool
import os
import json
from datetime import datetime
from utils.store import save_json_store, load_json_store


async def check_missing_packages_standalone(sandbox: AsyncSandbox) -> list[str]:
    """Deterministic package check — no LLM, just shell commands.

    Returns a list of missing package names (empty if all deps are installed).
    """
    try:
        result = await sandbox.commands.run(
            "grep -roh \"from ['\\\"][^'\\\"]*['\\\"]\" src/ 2>/dev/null"
            " | sed \"s/from ['\\\"]//;s/['\\\"]//\" | grep -v '^\\.'"
            " | cut -d/ -f1 | sort -u",
            cwd="/home/user/react-app",
            timeout=15,
        )
        imported = {p.strip() for p in result.stdout.strip().split("\n") if p.strip()}

        pkg_result = await sandbox.commands.run(
            "node -e \"const p=require('./package.json'); console.log(Object.keys(p.dependencies||{}).join('\\n'))\"",
            cwd="/home/user/react-app",
            timeout=10,
        )
        installed = {p.strip() for p in pkg_result.stdout.strip().split("\n") if p.strip()}

        builtin = {"react", "react-dom"}
        return sorted(imported - installed - builtin)

    except Exception as e:
        print(f"check_missing_packages_standalone failed: {e}")
        return []


def create_tools(
    sandbox: AsyncSandbox,
    event_queue: asyncio.Queue,
    project_id: str = None,
    files_tracker: list = None,
    mode: str = "first_build",
):
    """Create tools for a specific agent mode.

    Modes:
      - "first_build":  create_file, execute_command, write_multiple_files  (3 tools)
      - "follow_up":    + read_file, list_directory, get_context, save_context  (7 tools)
      - "fixer":        read_file, create_file, execute_command  (3 tools)
    """

    def safe_send_event(queue: asyncio.Queue, data: dict) -> bool:
        try:
            if queue:
                queue.put_nowait(data)
            return True
        except Exception as e:
            print(f"safe_send_event failed: {e}")
            return False

    # ── Core tools ───────────────────────────────────────────

    @tool
    async def create_file(file_path: str, content: str) -> str:
        """Create/overwrite a file. Path relative to root, e.g. "src/App.jsx"."""
        try:
            full_path = os.path.join("/home/user/react-app", file_path)
            await sandbox.files.write(full_path, content)
            if files_tracker is not None:
                files_tracker.append(file_path)
            safe_send_event(event_queue, {"e": "file_created", "message": f"Created {file_path}"})
            return f"File {file_path} created successfully."
        except Exception as e:
            safe_send_event(event_queue, {"e": "file_error", "message": f"Failed to create {file_path}: {str(e)}"})
            return f"Failed to create file {file_path}: {str(e)}"

    @tool
    async def read_file(file_path: str) -> str:
        """Read a file. Path relative to root, e.g. "src/App.jsx"."""
        try:
            full_path = os.path.join("/home/user/react-app", file_path)
            content = await sandbox.files.read(full_path)
            safe_send_event(event_queue, {"e": "file_read", "message": f"Read {file_path}"})
            return f"Content from {file_path}:\n{content}"
        except Exception as e:
            safe_send_event(event_queue, {"e": "file_error", "message": f"Failed to read {file_path}: {str(e)}"})
            return f"Failed to read file {file_path}: {str(e)}"

    # Track npm install calls to prevent duplicates
    npm_install_called = {"done": False}

    @tool
    async def execute_command(command: str) -> str:
        """Run a shell command in react-app dir."""
        try:
            # Prevent duplicate npm install in builder mode
            if mode == "first_build" and command.strip().startswith("npm install"):
                if npm_install_called["done"]:
                    return "SKIPPED: npm install already ran. Dependencies are installed."
                npm_install_called["done"] = True

            safe_send_event(event_queue, {"e": "command_started", "command": command})
            result = await sandbox.commands.run(command, cwd="/home/user/react-app", timeout=120)

            status = "command_executed" if result.exit_code == 0 else "command_failed"
            safe_send_event(event_queue, {
                "e": status, "command": command,
                "exit_code": result.exit_code,
                "stdout": result.stdout[:500] if result.stdout else "",
                "stderr": result.stderr[:500] if result.stderr else "",
            })

            if result.exit_code == 0:
                return f"OK: {result.stdout[:500]}"
            else:
                return f"FAIL (exit {result.exit_code}): {result.stderr[:500]}"

        except Exception as e:
            safe_send_event(event_queue, {"e": "command_error", "command": command, "message": str(e)})
            return f"ERROR: {str(e)}"

    @tool
    async def write_multiple_files(files: list[dict]) -> str:
        """Batch-write files. Each item: {"path":"src/...","data":"content"}. Preferred for multiple files."""
        try:
            invalid = [i for i, f in enumerate(files) if "path" not in f or "data" not in f]
            if invalid:
                return (
                    f"ERROR: files[{invalid}] missing required 'path' or 'data' key. "
                    "Each entry must be {\"path\": \"src/...\", \"data\": \"...content...\"}."
                )

            file_objects = [
                {
                    "path": os.path.join("/home/user/react-app", f["path"]),
                    "data": f["data"],
                }
                for f in files
            ]

            await sandbox.files.write_files(file_objects)

            file_names = [f["path"] for f in files]
            if files_tracker is not None:
                files_tracker.extend(file_names)

            safe_send_event(event_queue, {
                "e": "files_created",
                "message": f"Created {len(file_names)} files: {', '.join(file_names)}",
            })
            return f"Successfully created {len(file_names)} files: {', '.join(file_names)}"

        except Exception as e:
            safe_send_event(event_queue, {
                "e": "file_error",
                "message": f"Failed to create multiple files: {str(e)}",
            })
            return f"Failed to create multiple files: {str(e)}"

    @tool
    async def list_directory(path: str = ".") -> str:
        """List directory tree (excludes node_modules)."""
        try:
            cmd = f"tree -I 'node_modules|.*' {path}"
            result = await sandbox.commands.run(cmd, cwd="/home/user/react-app", timeout=30)
            safe_send_event(event_queue, {"e": "command_executed", "command": cmd})
            if result.exit_code == 0:
                return f"Directory structure:\n{result.stdout}"
            return f"Failed: {result.stderr}"
        except Exception as e:
            return f"Failed to list directory: {str(e)}"

    @tool
    def get_context() -> str:
        """Retrieve saved project context from previous sessions."""
        if not project_id:
            return "No project ID available"
        try:
            context = load_json_store(project_id, "context.json")
            if not context:
                return "No previous context found."

            result = "=== PROJECT CONTEXT ===\n\n"
            if context.get("semantic"):
                result += f"WHAT THIS PROJECT IS:\n{context['semantic']}\n\n"
            if context.get("procedural"):
                result += f"HOW THINGS WORK:\n{context['procedural']}\n\n"
            if context.get("episodic"):
                result += f"WHAT HAS BEEN DONE:\n{context['episodic']}\n\n"
            if context.get("files_created"):
                result += f"FILES CREATED: {len(context['files_created'])} files\n"
                result += f"   {', '.join(context['files_created'][:10])}"
                if len(context["files_created"]) > 10:
                    result += f" ... and {len(context['files_created']) - 10} more"
                result += "\n\n"
            if context.get("conversation_history"):
                result += "CONVERSATION HISTORY:\n"
                for i, conv in enumerate(context["conversation_history"][-5:], 1):
                    status = "[OK]" if conv.get("success") else "[FAIL]"
                    result += f"   {i}. {status} {conv.get('user_prompt', '')[:80]}\n"
            return result
        except Exception as e:
            return f"Failed to retrieve context: {str(e)}"

    @tool
    def save_context(semantic: str, procedural: str = "", episodic: str = "") -> str:
        """Save project context. semantic=what, procedural=how, episodic=done."""
        if not project_id:
            return "No project ID available"
        try:
            existing = load_json_store(project_id, "context.json")
            context = {
                "semantic": semantic or existing.get("semantic", ""),
                "procedural": procedural or existing.get("procedural", ""),
                "episodic": episodic or existing.get("episodic", ""),
                "last_updated": datetime.now().isoformat(),
                "files_created": existing.get("files_created", []),
                "conversation_history": existing.get("conversation_history", []),
            }
            save_json_store(project_id, "context.json", context)
            return f"Context saved for project {project_id}."
        except Exception as e:
            return f"Failed to save context: {str(e)}"

    # ── Assemble tool list by mode ───────────────────────────

    if mode == "first_build":
        return [create_file, execute_command, write_multiple_files]

    elif mode == "follow_up":
        return [
            create_file, read_file, execute_command,
            list_directory, write_multiple_files,
            get_context, save_context,
        ]

    elif mode == "fixer":
        return [read_file, create_file]

    else:
        raise ValueError(f"Unknown tool mode: {mode}")
