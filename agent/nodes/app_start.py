"""
Node 6: APP START — deterministic (no LLM).

Ensures the Vite dev server is running, serving HTTP 200,
and has no module-level or runtime errors.
"""

import asyncio
import re

from langchain_core.runnables import RunnableConfig

from ..graph_state import GraphState
from .helpers import safe_send_event, NodeTimer


# Patterns that indicate errors in Vite page HTML or logs
_ERROR_PATTERNS = [
    r"vite-error-overlay",
    r"Internal Server Error",
    r"SyntaxError:",
    r"ReferenceError:",
    r"TypeError:",
    r"Cannot find module",
    r"Module not found",
    r"Failed to resolve import",
    r"does not provide an export named",
    r"is not defined",
    r"Uncaught",
    r"pre-transform error",
]
_ERROR_RE = re.compile("|".join(_ERROR_PATTERNS), re.IGNORECASE)


async def _check_runtime_errors(sandbox, path: str) -> list[str]:
    """Fetch the Vite page and logs, looking for runtime/module errors.

    Returns a list of error descriptions (empty = all clear).
    """
    errors: list[str] = []

    # 1. Fetch the actual page body and look for error overlay / module errors
    try:
        body_result = await sandbox.commands.run(
            "curl -s --max-time 10 http://localhost:5173",
            cwd=path,
            timeout=15,
        )
        body = body_result.stdout or ""
        if _ERROR_RE.search(body):
            # Extract a short snippet around the match for context
            match = _ERROR_RE.search(body)
            start = max(0, match.start() - 100)
            end = min(len(body), match.end() + 200)
            snippet = body[start:end].strip()
            # Clean HTML tags for readability
            snippet = re.sub(r"<[^>]+>", " ", snippet)
            snippet = re.sub(r"\s+", " ", snippet).strip()
            errors.append(f"Page contains error: {snippet[:300]}")
            print("App start: Runtime error detected in page HTML")
    except Exception as e:
        print(f"App start: Failed to fetch page body: {e}")

    # 2. Check Vite logs for errors
    try:
        log_result = await sandbox.commands.run(
            "tail -50 /tmp/vite.log 2>/dev/null || true",
            cwd=path,
            timeout=10,
        )
        log_text = log_result.stdout or ""
        if log_text:
            for line in log_text.splitlines():
                if _ERROR_RE.search(line):
                    clean_line = line.strip()[:200]
                    errors.append(f"Vite log error: {clean_line}")
                    print(f"App start: Error in Vite log: {clean_line}")
                    break  # One error from logs is enough context
    except Exception as e:
        print(f"App start: Failed to read Vite logs: {e}")

    return errors


async def app_start_node(state: GraphState, config: RunnableConfig) -> dict:
    """Ensure the Vite dev server is running, serving, and error-free."""
    timer = NodeTimer("app_start")
    configurable = config.get("configurable", {})
    event_queue = configurable.get("event_queue")
    sandbox = configurable.get("sandbox")

    safe_send_event(event_queue, {"e": "app_check_started", "message": "Starting your app and running final checks..."})

    try:
        if not sandbox:
            raise Exception("Sandbox not available")

        path = "/home/user/react-app"
        runtime_errors = []

        # Check essential files
        main_files = ["src/App.jsx", "src/main.jsx", "package.json"]
        missing_files = []
        for file_path in main_files:
            try:
                await sandbox.files.read(f"{path}/{file_path}")
            except Exception:
                missing_files.append(file_path)

        if missing_files:
            runtime_errors.append(f"Missing essential files: {', '.join(missing_files)}")
        else:
            # Determine if we need a fresh Vite restart.
            # After fixer runs, we must restart Vite to clear stale error logs.
            fixer_ran = state.get("fixer_retries", 0) > 0

            port_check = await sandbox.commands.run(
                "ss -tlnp 2>/dev/null | grep -q ':5173' && echo 'port_open' || echo 'port_closed'",
                cwd=path,
            )
            port_open = "port_open" in port_check.stdout

            need_restart = not port_open or fixer_ran

            if need_restart:
                if port_open:
                    # Kill existing Vite so we get fresh logs
                    # Use lsof to find the exact PID to avoid pkill matching itself
                    await sandbox.commands.run(
                        "kill $(lsof -ti :5173) 2>/dev/null || true",
                        cwd=path, timeout=5,
                    )
                    await asyncio.sleep(1)
                    print("App start: Killed existing Vite to get fresh logs after fixer")
                else:
                    print("App start: Port 5173 not listening")

                # Clear old logs and start fresh
                await sandbox.commands.run("rm -f /tmp/vite.log", cwd=path, timeout=5)
                await sandbox.commands.run(
                    "nohup npm run dev -- --host 0.0.0.0 > /tmp/vite.log 2>&1 &",
                    cwd=path,
                )
                for attempt in range(15):
                    await asyncio.sleep(2)
                    poll = await sandbox.commands.run(
                        "ss -tlnp 2>/dev/null | grep -q ':5173' && echo 'ready' || echo 'waiting'",
                        cwd=path,
                    )
                    if "ready" in poll.stdout:
                        print(f"App start: Vite ready after {(attempt + 1) * 2}s")
                        break
                else:
                    print("App start: Vite did not start within 30s")
            else:
                print("App start: Port 5173 already open (no fixer ran, keeping existing)")

            # Verify HTTP response
            vite_result = await sandbox.commands.run(
                "curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://localhost:5173",
                cwd=path,
            )
            http_code = vite_result.stdout.strip()
            if http_code == "200":
                print("App start: Vite is responding (HTTP 200)")
            else:
                print(f"App start: Vite returned HTTP {http_code}")
                runtime_errors.append(f"Vite dev server returned HTTP {http_code}")

            # ── Runtime error detection ──
            # Wait briefly for React to mount, then check for errors
            if not runtime_errors:
                await asyncio.sleep(2)
                page_errors = await _check_runtime_errors(sandbox, path)
                runtime_errors.extend(page_errors)

        success = len(runtime_errors) == 0
        safe_send_event(event_queue, {
            "e": "app_check_complete",
            "errors": runtime_errors,
            "message": "App is ready" if success else f"App check found {len(runtime_errors)} issues",
        })

        error_str = "; ".join(runtime_errors) if runtime_errors else None

        log_entry = timer.stop(errors=runtime_errors)
        result = {
            "success": success,
            "current_node": "app_start",
            "error_message": error_str,
            "execution_log": [log_entry],
        }
        # Put runtime errors into build_errors so the fixer can read them
        if error_str:
            result["build_errors"] = error_str
        return result

    except Exception as e:
        error_msg = f"App start error: {str(e)}"
        print(error_msg)
        safe_send_event(event_queue, {"e": "app_check_error", "message": error_msg})

        log_entry = timer.stop(status="error", error=error_msg)
        return {
            "success": False,
            "current_node": "app_start",
            "error_message": error_msg,
            "execution_log": [log_entry],
        }
