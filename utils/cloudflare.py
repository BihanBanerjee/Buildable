"""
Cloudflare Pages deploy — runs `npm run build` + `wrangler pages deploy` inside E2B sandbox.
"""

import os
import re

from e2b_code_interpreter import AsyncSandbox


def _sanitize_project_name(name: str) -> str:
    """Sanitize project name for Cloudflare Pages (lowercase, alphanumeric + dashes, max 58 chars)."""
    safe = re.sub(r"[^a-z0-9-]", "-", name.lower())
    safe = re.sub(r"-+", "-", safe).strip("-")
    return safe[:58] or "buildable-app"


async def deploy_to_cloudflare(sandbox: AsyncSandbox, project_name: str) -> dict:
    """Deploy a project to Cloudflare Pages via Wrangler inside the E2B sandbox.

    Returns {"success": True, "url": "https://..."} or {"success": False, "error": "..."}.
    """
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN", "")

    if not account_id or not api_token:
        return {"success": False, "error": "Cloudflare credentials not configured. Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env."}

    safe_name = _sanitize_project_name(project_name)
    path = "/home/user/react-app"
    env_prefix = f"CLOUDFLARE_ACCOUNT_ID='{account_id}' CLOUDFLARE_API_TOKEN='{api_token}'"

    # Step 1: Build the project
    try:
        build_result = await sandbox.commands.run("npm run build", cwd=path, timeout=60)
        if build_result.exit_code != 0:
            error = build_result.stderr or build_result.stdout or "Unknown build error"
            return {"success": False, "error": f"Build failed: {error[:500]}"}
    except Exception as e:
        return {"success": False, "error": f"Build timed out or failed: {str(e)[:200]}"}

    # Step 2: Create Cloudflare Pages project (idempotent — ignores "already exists")
    try:
        create_result = await sandbox.commands.run(
            f"{env_prefix} npx wrangler pages project create '{safe_name}' --production-branch main",
            cwd=path,
            timeout=30,
        )
        output = (create_result.stderr or "") + (create_result.stdout or "")
        already_exists = any(
            marker in output.lower()
            for marker in ["already exists", "8000002", "8000000", "resource already"]
        )
        if create_result.exit_code != 0 and not already_exists:
            print(f"Wrangler project create warning (proceeding anyway): {output[:200]}")
    except Exception as e:
        print(f"Wrangler project create failed (proceeding anyway): {e}")

    # Step 3: Deploy to Cloudflare Pages
    try:
        deploy_result = await sandbox.commands.run(
            f"{env_prefix} npx wrangler pages deploy dist --project-name='{safe_name}'",
            cwd=path,
            timeout=120,
        )
        if deploy_result.exit_code != 0:
            error = deploy_result.stderr or deploy_result.stdout or "Unknown deploy error"
            return {"success": False, "error": f"Deploy failed: {error[:500]}"}
    except Exception as e:
        return {"success": False, "error": f"Deploy timed out or failed: {str(e)[:200]}"}

    # Step 4: Parse deployed URL from wrangler output
    output = (deploy_result.stdout or "") + (deploy_result.stderr or "")
    url_match = re.search(r"https://[^\s]+\.pages\.dev", output)
    url = url_match.group(0) if url_match else f"https://{safe_name}.pages.dev"

    print(f"Deployed to Cloudflare Pages: {url}")
    return {"success": True, "url": url}
