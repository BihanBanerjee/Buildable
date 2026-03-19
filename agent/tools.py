from e2b_code_interpreter import AsyncSandbox
import asyncio
from langchain_core.tools import tool
import os
import json
from datetime import datetime
from utils.store import save_json_store, load_json_store


def create_tools_with_context(
    sandbox: AsyncSandbox, event_queue: asyncio.Queue, project_id: str = None,
    validation_results: dict = None, files_tracker: list = None,
    include_test_build: bool = True, first_build: bool = False,
):
    """Create tools with sandbox and event queue context"""
    def safe_send_event(queue: asyncio.Queue, data: dict) -> bool:
        """Safely send event to queue; return False if send fails."""
        try:
            if queue:
                queue.put_nowait(data)
            return True
        except Exception as e:
            # Log and return False so callers can stop trying to send further messages.
            print(f"safe_send_event failed: {e}")
            return False

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
            safe_send_event(event_queue, {"e": "file_error", "message": f"Failed to create {file_path}: {str(e)}",})
            return f"Failed to create file {file_path}: {str(e)}"

    @tool
    async def read_file(file_path: str) -> str:
        """Read a file. Path relative to root, e.g. "src/App.jsx"."""
        try:
            full_path = os.path.join("/home/user/react-app", file_path)
            content = await sandbox.files.read(full_path)
            safe_send_event(event_queue, {"e": "file_read", "message": f"Read content from {file_path}"})
            return f"Content from {file_path}:\n{content}"
        except Exception as e:
            safe_send_event(event_queue, {"e": "file_error", "message": f"Failed to read {file_path}: {str(e)}"})
            return f"Failed to read file {file_path}: {str(e)}"

    @tool
    async def delete_file(file_path: str) -> str:
        """Delete a file. Path relative to root."""
        try:
            full_path = os.path.join("/home/user/react-app", file_path)
            await sandbox.files.remove(full_path)
            safe_send_event(event_queue, {"e": "file_deleted", "message": f"Deleted {file_path}"})
            return f"File {file_path} deleted successfully."
        except Exception as e:
            safe_send_event(event_queue, {"e": "file_error", "message": f"Failed to delete {file_path}: {str(e)}",})
            return f"Failed to delete file {file_path}: {str(e)}"

    @tool
    async def execute_command(command: str) -> str:
        """Run a shell command in react-app dir."""
        try:
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
    async def test_build() -> str:
        """Run Vite build to check for compile errors."""
        try:
            path = "/home/user/react-app"
            safe_send_event(event_queue, {"e": "build_test_started", "message": "Testing build..."})

            # Clean Vite cache only (npm install already ran during build)
            await sandbox.commands.run("rm -rf node_modules/.vite-temp", cwd=path, timeout=10)

            res = await sandbox.commands.run("npx vite build --mode development", cwd=path, timeout=120)

            if res.exit_code == 0:
                safe_send_event(event_queue, {"e": "build_test_success", "message": "Build passed"})
                return f"Build PASSED.\n{res.stdout[:300]}"
            else:
                out = res.stderr or res.stdout
                safe_send_event(event_queue, {"e": "build_test_failed", "message": "Build failed", "error": out[:500]})
                return f"Build FAILED (exit {res.exit_code}).\n{out[:800]}"

        except Exception as e:
            return f"Build test error: {str(e)}"

    @tool
    async def write_multiple_files(files: list[dict]) -> str:
        """Batch-write files. Each item: {"path":"src/...","data":"content"}. Preferred for multiple files."""
        try:
            # Validate each entry has the required keys before touching the sandbox
            invalid = [i for i, f in enumerate(files) if "path" not in f or "data" not in f]
            if invalid:
                return (
                    f"ERROR: files[{invalid}] missing required 'path' or 'data' key. "
                    "Each entry must be {\"path\": \"src/...\", \"data\": \"...content...\"}."
                )

            # Convert to the absolute paths E2B expects
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

            safe_send_event(
                event_queue,
                {
                    "e": "files_created",
                    "message": f"Created {len(file_names)} files: {', '.join(file_names)}",
                },
            )
            return f"Successfully created {len(file_names)} files: {', '.join(file_names)}"

        except Exception as e:
            safe_send_event(
                event_queue,
                {
                    "e": "file_error",
                    "message": f"Failed to create multiple files: {str(e)}",
                },
            )
            return f"Failed to create multiple files: {str(e)}"

    @tool
    def get_context() -> str:
        """Retrieve saved project context from previous sessions."""
        if not project_id:
            return "No project ID available - context cannot be retrieved"

        try:
            context = load_json_store(project_id, "context.json")

            if not context:
                return "No previous context found for this project. This appears to be a new project."

            # Format the context for display
            result = "=== PROJECT CONTEXT ===\n\n"

            if context.get("semantic"):
                result += "📋 WHAT THIS PROJECT IS:\n"
                result += f"{context['semantic']}\n\n"

            if context.get("procedural"):
                result += "HOW THINGS WORK:\n"
                result += context["procedural"] + "\n\n"

            if context.get("episodic"):
                result += "WHAT HAS BEEN DONE:\n"
                result += f"{context['episodic']}\n\n"

            if context.get("files_created"):
                result += f"📁 FILES CREATED: {len(context['files_created'])} files\n"
                result += f"   {', '.join(context['files_created'][:10])}"
                if len(context["files_created"]) > 10:
                    result += f" ... and {len(context['files_created']) - 10} more"
                result += "\n\n"

            if context.get("conversation_history"):
                result += "CONVERSATION HISTORY:\n"
                for i, conv in enumerate(context["conversation_history"][-5:], 1):
                    status = "[SUCCESS]" if conv.get("success") else "[FAILED]"
                    result += f"   {i}. {status} {conv.get('user_prompt', 'Unknown')[:80]}...\n"
                result += "\n"

            if context.get("last_updated"):
                result += f"Last Updated: {context['last_updated']}\n"

            return result

        except Exception as e:
            return f"Failed to retrieve context: {str(e)}"

    @tool
    def save_context(semantic: str, procedural: str = "", episodic: str = "") -> str:
        """Save project context. semantic=what, procedural=how, episodic=done."""
        if not project_id:
            return "No project ID available - context cannot be saved"

        try:
            # Load existing context to preserve information
            existing_context = load_json_store(project_id, "context.json")

            # Update context with new information
            context = {
                "semantic": semantic or existing_context.get("semantic", ""),
                "procedural": procedural or existing_context.get("procedural", ""),
                "episodic": episodic or existing_context.get("episodic", ""),
                "last_updated": datetime.now().isoformat(),
                "files_created": existing_context.get("files_created", []),
                "conversation_history": existing_context.get(
                    "conversation_history", []
                ),
            }

            # Save to store
            save_json_store(project_id, "context.json", context)

            return f"Context saved successfully for project {project_id}. This information will be available in future sessions."

        except Exception as e:
            return f"Failed to save context: {str(e)}"

    @tool
    async def check_missing_packages() -> str:
        """Scan imports vs package.json; report missing npm deps."""
        try:
            # Single shell pipeline: extract all third-party imports + read package.json deps
            # This replaces N individual file reads with ONE sandbox command.
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

            # Built-ins that are always available
            builtin = {"react", "react-dom"}
            missing = sorted(imported - installed - builtin)

            if missing:
                safe_send_event(event_queue, {"e": "missing_dependencies", "packages": missing})
                return f"MISSING: {', '.join(missing)}\nRun: npm install {' '.join(missing)}"
            else:
                return "All dependencies installed. No missing packages."

        except Exception as e:
            return f"Dependency check failed: {str(e)}"

    # --- Assemble tool list based on caller ---
    if validation_results is not None:
        # Validator only needs: read, fix, run commands, check deps, and report.
        # Fewer tools = smaller schema = fewer tokens per LLM call.
        @tool
        def report_validation_result(errors: list[str], summary: str) -> str:
            """REQUIRED final action. errors=[] if clean, otherwise list remaining issues. summary=one-line description."""
            validation_results["errors"] = errors
            validation_results["summary"] = summary
            safe_send_event(
                event_queue,
                {"e": "validation_report", "errors": errors, "summary": summary},
            )
            return f"Validation report saved. Remaining errors: {len(errors)}."

        return [
            read_file,
            create_file,
            execute_command,
            check_missing_packages,
            report_validation_result,
        ]

    # First build: minimal tool set (fewer tools = smaller schema = fewer tokens per LLM call)
    if first_build:
        tools = [
            create_file,
            read_file,
            execute_command,
            write_multiple_files,
            check_missing_packages,
        ]
    else:
        # Follow-up builds get the full tool set
        tools = [
            create_file,
            read_file,
            execute_command,
            delete_file,
            list_directory,
            write_multiple_files,
            get_context,
            save_context,
            check_missing_packages,
        ]

    if include_test_build:
        tools.insert(3, test_build)

    return tools
