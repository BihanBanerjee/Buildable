"""
Node 4: BUILD CHECKPOINT — deterministic vite build (no LLM).

Runs `npx vite build`, auto-installs missing packages on fixer retries.
"""

import asyncio
import traceback

from langchain_core.runnables import RunnableConfig

from ..graph_state import GraphState
from ..tools import check_missing_packages_standalone
from .helpers import safe_send_event, NodeTimer


async def build_checkpoint_node(state: GraphState, config: RunnableConfig) -> dict:
    """Deterministic build check. Runs vite build + auto-installs missing packages.

    On first pass after scaffold, skips npm install (scaffold already did it).
    On fixer retries, runs full package check.
    """
    timer = NodeTimer("build_checkpoint")
    configurable = config.get("configurable", {})
    event_queue = configurable.get("event_queue")
    sandbox = configurable.get("sandbox")

    safe_send_event(event_queue, {"e": "code_validator_started", "message": "Validating and fixing any issues..."})

    try:
        if not sandbox:
            raise Exception("Sandbox not available")

        path = "/home/user/react-app"
        fixer_retries = state.get("fixer_retries", 0)

        # Only check for missing packages on fixer retries (scaffold handles first install)
        if fixer_retries > 0:
            missing = await check_missing_packages_standalone(sandbox)
            if missing:
                print(f"Build checkpoint: installing missing packages: {missing}")
                safe_send_event(event_queue, {"e": "missing_dependencies", "packages": missing})
                install_cmd = f"npm install {' '.join(missing)}"
                await sandbox.commands.run(install_cmd, cwd=path, timeout=120)

        # Clean Vite cache
        await sandbox.commands.run("rm -rf node_modules/.vite-temp", cwd=path, timeout=10)

        # Run vite build
        result = await asyncio.wait_for(
            sandbox.commands.run("npx vite build --mode development 2>&1", cwd=path, timeout=120),
            timeout=130,
        )

        build_passed = result.exit_code == 0
        build_errors = "" if build_passed else (result.stderr or result.stdout or "Unknown build error")

        if build_passed:
            print("Build checkpoint: PASSED")
            safe_send_event(event_queue, {"e": "validation_success", "message": "Build passed"})
        else:
            error_lines = build_errors.strip().split("\n")
            build_errors = "\n".join(error_lines[-40:])
            print(f"Build checkpoint: FAILED\n{build_errors[:500]}")
            safe_send_event(event_queue, {"e": "build_test_failed", "message": "Build failed", "error": build_errors[:500]})

        safe_send_event(event_queue, {"e": "code_validator_complete", "message": "Validation complete"})

        log_entry = timer.stop(status="passed" if build_passed else "failed")
        return {
            "build_passed": build_passed,
            "build_errors": build_errors,
            "current_node": "build_checkpoint",
            "execution_log": [log_entry],
        }

    except Exception as e:
        error_msg = f"Build checkpoint error: {str(e)}"
        print(error_msg)
        safe_send_event(event_queue, {"e": "code_validator_complete", "message": error_msg})

        log_entry = timer.stop(status="error", error=error_msg)
        return {
            "build_passed": False,
            "build_errors": error_msg,
            "current_node": "build_checkpoint",
            "execution_log": [log_entry],
        }
