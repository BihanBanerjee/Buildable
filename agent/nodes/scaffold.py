"""
Node 2: SCAFFOLD — deterministic setup (no LLM).

Writes base template files and installs npm dependencies.
App.jsx, pages, and components are created by the builder node.
"""

import asyncio
import traceback

from langchain_core.runnables import RunnableConfig

from ..graph_state import GraphState
from ..base_template import BASE_TEMPLATE
from .helpers import safe_send_event, NodeTimer


# ─────────────────────────────────────────────────────────────
# Node entry point
# ─────────────────────────────────────────────────────────────

async def scaffold_node(state: GraphState, config: RunnableConfig) -> dict:
    """Deterministic scaffolding — write base template files and install deps.

    Zero LLM tokens. Runs in ~10s (mostly npm install).
    App.jsx, pages, and components are created by the builder node.
    """
    timer = NodeTimer("scaffold")
    configurable = config.get("configurable", {})
    event_queue = configurable.get("event_queue")
    sandbox = configurable.get("sandbox")

    # ── Follow-up path: skip scaffold, hand off to builder ──
    if not state.get("is_first_message", True):
        safe_send_event(event_queue, {"e": "builder_started", "message": "Generating code for your application..."})
        log_entry = timer.stop(status="skipped (follow-up)")
        return {
            "scaffold_complete": True,
            "files_created": [],
            "current_node": "scaffold",
            "execution_log": [log_entry],
        }

    # ── First-build path ──
    try:
        if not sandbox:
            raise Exception("Sandbox not available")

        plan = state.get("plan", {})
        dependencies = plan.get("dependencies", [])
        path = "/home/user/react-app"
        files_created = []

        safe_send_event(event_queue, {"e": "builder_started", "message": "Generating code for your application..."})

        # Step 1: Write base template files in parallel (locked infra files)
        async def write_base_file(file_path: str, content: str):
            await sandbox.files.write(f"{path}/{file_path}", content)

        await asyncio.gather(*[
            write_base_file(fp, content)
            for fp, content in BASE_TEMPLATE.items()
        ])
        files_created.extend(BASE_TEMPLATE.keys())
        safe_send_event(event_queue, {
            "e": "tool_completed",
            "tool_name": "scaffold",
            "tool_output": f"Wrote {len(BASE_TEMPLATE)} base template files",
        })

        # Step 2: Install plan dependencies (with --legacy-peer-deps fallback)
        # npm install from base template package.json first
        safe_send_event(event_queue, {
            "e": "tool_started",
            "tool_name": "execute_command",
            "tool_input": {"command": "npm install"},
        })
        try:
            result = await sandbox.commands.run("npm install", cwd=path, timeout=120)
            install_ok = result.exit_code == 0
        except Exception:
            install_ok = False

        if not install_ok:
            try:
                await sandbox.commands.run("npm install --legacy-peer-deps", cwd=path, timeout=120)
            except Exception as e:
                print(f"Scaffold base npm install failed: {e}")

        # Install extra plan dependencies one-by-one so a bad package doesn't block others
        for dep in dependencies:
            try:
                result = await sandbox.commands.run(f"npm install {dep}", cwd=path, timeout=60)
                if result.exit_code != 0:
                    # Retry with --legacy-peer-deps
                    result = await sandbox.commands.run(f"npm install --legacy-peer-deps {dep}", cwd=path, timeout=60)
                    if result.exit_code != 0:
                        print(f"Scaffold: failed to install {dep} (skipping)")
                    else:
                        print(f"Scaffold: installed {dep} with --legacy-peer-deps")
            except Exception as e:
                print(f"Scaffold: failed to install {dep}: {str(e)[:100]} (skipping)")

        safe_send_event(event_queue, {
            "e": "tool_completed",
            "tool_name": "execute_command",
            "tool_output": "Dependencies installed",
        })

        # App.jsx, pages, and components are created by the builder (not scaffold).
        # The builder includes App.jsx with context providers in its write_multiple_files call.
        print(f"Scaffold complete: {len(files_created)} base files, {len(dependencies)} deps")

        log_entry = timer.stop(files=files_created)
        return {
            "scaffold_complete": True,
            "files_created": files_created,
            "current_node": "scaffold",
            "execution_log": [log_entry],
        }

    except Exception as e:
        error_msg = f"Scaffold error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        safe_send_event(event_queue, {"e": "builder_started", "message": "Generating code for your application..."})

        log_entry = timer.stop(status="error", error=error_msg)
        return {
            "scaffold_complete": False,
            "current_node": "scaffold",
            "error_message": error_msg,
            "execution_log": [log_entry],
        }
