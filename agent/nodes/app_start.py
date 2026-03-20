"""
Node 6: APP START — deterministic (no LLM).

Ensures the Vite dev server is running and serving HTTP 200.
"""

import asyncio
import traceback

from langchain_core.runnables import RunnableConfig

from ..graph_state import GraphState
from .helpers import safe_send_event, NodeTimer


async def app_start_node(state: GraphState, config: RunnableConfig) -> dict:
    """Ensure the Vite dev server is running and serving."""
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
            # Check if Vite is listening on port 5173
            port_check = await sandbox.commands.run(
                "ss -tlnp 2>/dev/null | grep -q ':5173' && echo 'port_open' || echo 'port_closed'",
                cwd=path,
            )
            if "port_closed" in port_check.stdout:
                print("App start: Port 5173 not listening — restarting Vite")
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
                print("App start: Port 5173 already open")

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

        success = len(runtime_errors) == 0
        safe_send_event(event_queue, {
            "e": "app_check_complete",
            "errors": runtime_errors,
            "message": "App is ready" if success else f"App check found {len(runtime_errors)} issues",
        })

        log_entry = timer.stop(errors=runtime_errors)
        return {
            "success": success,
            "current_node": "app_start",
            "error_message": "; ".join(runtime_errors) if runtime_errors else None,
            "execution_log": [log_entry],
        }

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
