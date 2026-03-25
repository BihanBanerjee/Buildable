"""
Sandbox lifecycle — create, check, update, and validate E2B sandboxes.

Each project gets an isolated Linux sandbox with a React + Vite environment.
The sandbox runs for up to 1 hour (3600s timeout).
"""

import asyncio
import os

from e2b_code_interpreter import AsyncSandbox
from typing import Callable


SANDBOX_TIMEOUT = 3600  # 1 hour
TEMPLATE_ID = os.getenv("E2B_TEMPLATE_ID", None)
BASE_PATH = "/home/user/react-app"


async def create_sandbox(
    project_files: dict[str, str],
    on_log: Callable[[str], None] | None = None,
) -> dict:
    """Create E2B sandbox, write files, install deps, start Vite.

    Args:
        project_files: {relative_path: content} — complete project.
        on_log: Optional callback for streaming log messages.

    Returns:
        {"sandbox_id": str, "url": str, "sandbox": AsyncSandbox}
    """
    def log(msg: str):
        print(msg)
        if on_log:
            on_log(msg)

    # Create sandbox
    if TEMPLATE_ID:
        log(f"Creating sandbox with template {TEMPLATE_ID}...")
        sbx = await AsyncSandbox.create(template=TEMPLATE_ID, timeout=SANDBOX_TIMEOUT)
    else:
        log("Creating sandbox...")
        sbx = await AsyncSandbox.create(timeout=SANDBOX_TIMEOUT)

    log(f"Sandbox created: {sbx.sandbox_id}")

    # Create directories (with base path prefix)
    dirs = set()
    for file_path in project_files.keys():
        parent = os.path.dirname(file_path)
        if parent and parent != "." and parent != "/":
            dirs.add(f"{BASE_PATH}/{parent}")

    if dirs:
        await asyncio.gather(*[
            sbx.commands.run(f"mkdir -p {d}")
            for d in dirs
        ])

    # Write files (with base path prefix)
    log("Writing project files to sandbox...")
    await asyncio.gather(*[
        sbx.files.write(f"{BASE_PATH}/{file_path}", content)
        for file_path, content in project_files.items()
    ])
    log(f"Wrote {len(project_files)} files")

    # Install dependencies
    log("Installing dependencies...")
    await sbx.commands.run("npm install", cwd=BASE_PATH, timeout=120)

    # Start dev server (background)
    log("Starting dev server...")
    await sbx.commands.run("npm run dev", cwd=BASE_PATH, background=True)

    # Generate preview URL
    host = sbx.get_host(port=5173)
    url = f"https://{host}"
    log(f"Preview running at: {url}")

    return {"sandbox_id": sbx.sandbox_id, "url": url, "sandbox": sbx}


async def is_sandbox_alive(sandbox_id: str) -> bool:
    """Check if sandbox is alive (echo test, 5s timeout)."""
    try:
        sbx = await AsyncSandbox.reconnect(sandbox_id)
        await sbx.commands.run("echo alive", timeout=5)
        return True
    except Exception as e:
        print(f"Sandbox {sandbox_id} is not alive: {e}")
        return False


async def update_sandbox_files(
    sandbox: AsyncSandbox,
    files: dict[str, str],
) -> None:
    """Write files to sandbox. Restart Vite if not running.

    Args:
        sandbox: Connected AsyncSandbox instance.
        files: {relative_path: content} to write.
    """
    # Create directories (with base path prefix)
    dirs = set()
    for file_path in files.keys():
        parent = os.path.dirname(file_path)
        if parent and parent != "." and parent != "/":
            dirs.add(f"{BASE_PATH}/{parent}")

    if dirs:
        await asyncio.gather(*[
            sandbox.commands.run(f"mkdir -p {d}")
            for d in dirs
        ])

    # Write files (with base path prefix)
    await asyncio.gather(*[
        sandbox.files.write(f"{BASE_PATH}/{file_path}", content)
        for file_path, content in files.items()
    ])

    # Check if dev server is running
    check = await sandbox.commands.run(
        "lsof -i :5173 2>/dev/null || echo 'port_not_open'",
        timeout=5,
    )
    is_running = "port_not_open" not in check.stdout and check.stdout.strip()

    if not is_running:
        print("Dev server not running, restarting...")
        await sandbox.commands.run("npm install", cwd=BASE_PATH, timeout=120)
        await sandbox.commands.run("npm run dev", cwd=BASE_PATH, background=True)
        await asyncio.sleep(5)
        print("Dev server restarted.")


async def validate_sandbox_build(sandbox: AsyncSandbox) -> dict:
    """Run 'npm run build 2>&1' with 60s timeout.

    Returns:
        {"success": bool, "errors": str | None}
    """
    try:
        result = await asyncio.wait_for(
            sandbox.commands.run(
                "npm run build",
                cwd=BASE_PATH,
                timeout=60,
            ),
            timeout=70,
        )

        if result.exit_code == 0:
            print("Build validation: PASSED")
            return {"success": True, "errors": None}

        errors = (result.stdout or "") + "\n" + (result.stderr or "")
        errors = errors.strip() or "Unknown build error"
        # Keep last 40 lines
        error_lines = errors.split("\n")
        errors = "\n".join(error_lines[-40:])
        print(f"Build validation: FAILED\n{errors[:500]}")
        return {"success": False, "errors": errors}

    except Exception as e:
        # E2B may throw on non-zero exit — try to extract output from the exception
        error_msg = str(e) or "Unknown build error"
        stdout = getattr(e, 'stdout', '') or ''
        stderr = getattr(e, 'stderr', '') or ''
        if stdout or stderr:
            error_msg = (stdout + "\n" + stderr).strip()
        # Keep last 40 lines
        error_lines = error_msg.split("\n")
        error_msg = "\n".join(error_lines[-40:]) or "Build failed with unknown error"
        print(f"Build validation error:\n{error_msg[:500]}")
        return {"success": False, "errors": error_msg}
