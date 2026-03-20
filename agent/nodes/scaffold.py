"""
Node 2: SCAFFOLD — deterministic setup (no LLM).

Installs deps, generates App.jsx with routes, creates page stubs.
On follow-ups where the plan introduces new pages, auto-updates App.jsx routes.
"""

import traceback

from langchain_core.runnables import RunnableConfig

from ..graph_state import GraphState
from .helpers import safe_send_event, NodeTimer


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _page_name_to_route(page_name: str, index: int) -> str:
    """Convert a page name to a URL route. First page is always /."""
    if index == 0:
        return "/"
    route = page_name[0].lower()
    for ch in page_name[1:]:
        if ch.isupper():
            route += f"-{ch.lower()}"
        else:
            route += ch
    return f"/{route}"


def _generate_app_jsx(pages: list[str], components: list[str]) -> str:
    """Generate App.jsx with routes from the plan."""
    if not pages:
        pages = ["Home"]

    imports = [
        "import { BrowserRouter, Routes, Route } from 'react-router-dom'",
    ]
    for page in pages:
        imports.append(f"import {page} from './pages/{page}'")

    routes = []
    for i, page in enumerate(pages):
        route_path = _page_name_to_route(page, i)
        routes.append(f'        <Route path="{route_path}" element={{<{page} />}} />')

    nav_section = ""
    if len(pages) > 1:
        nav_links = []
        for i, page in enumerate(pages):
            route_path = _page_name_to_route(page, i)
            label = page.replace("Page", "")
            nav_links.append(
                f'            <a href="{route_path}" className="px-4 py-2 text-sm font-medium text-gray-300 hover:text-white hover:bg-gray-700 rounded-lg transition-colors">{label}</a>'
            )
        nav_section = f"""
      <nav className="bg-gray-900 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-2">
          <span className="text-white font-bold text-lg mr-4">App</span>
{chr(10).join(nav_links)}
        </div>
      </nav>"""

    return f"""import React from 'react'
{chr(10).join(imports)}

export default function App() {{
  return (
    <BrowserRouter>{nav_section}
      <Routes>
{chr(10).join(routes)}
      </Routes>
    </BrowserRouter>
  )
}}
"""



# ─────────────────────────────────────────────────────────────
# Follow-up scaffold: detect & register new pages
# ─────────────────────────────────────────────────────────────

async def _update_routes_for_new_pages(sandbox, plan: dict, event_queue) -> list[str]:
    """If the plan introduces pages not yet in App.jsx, regenerate routes.

    Returns list of newly-created stub page files.
    """
    path = "/home/user/react-app"
    new_files: list[str] = []

    try:
        # Read current App.jsx to find existing page imports
        current_app = await sandbox.files.read(f"{path}/src/App.jsx")
    except Exception:
        # No App.jsx — nothing to update
        return new_files

    planned_pages = plan.get("pages", [])
    if not planned_pages:
        return new_files

    # Detect which pages are already imported
    existing_pages: set[str] = set()
    for line in current_app.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") and "./pages/" in stripped:
            # e.g. import Settings from './pages/Settings'
            parts = stripped.split()
            if len(parts) >= 2:
                existing_pages.add(parts[1])

    new_pages = [p for p in planned_pages if p not in existing_pages]
    if not new_pages:
        return new_files

    print(f"Scaffold follow-up: adding {len(new_pages)} new page(s): {new_pages}")

    # Merge existing + new pages and regenerate App.jsx
    all_pages = list(existing_pages) + new_pages
    # Preserve original ordering: existing first (in their order), then new
    # We'll just regenerate with all planned pages since the plan is authoritative
    components = plan.get("components", [])
    app_jsx = _generate_app_jsx(planned_pages, components)
    await sandbox.files.write(f"{path}/src/App.jsx", app_jsx)

    safe_send_event(event_queue, {
        "e": "tool_completed",
        "tool_name": "scaffold",
        "tool_output": f"Updated App.jsx routes: added {', '.join(new_pages)}",
    })

    # Create stub files for new pages so imports don't break
    for page in new_pages:
        page_content = f"""import React from 'react'

export default function {page}() {{
  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <h1 className="text-3xl font-bold text-gray-900">{page}</h1>
    </div>
  )
}}
"""
        try:
            # Only write if file doesn't exist yet
            await sandbox.files.read(f"{path}/src/pages/{page}.jsx")
        except Exception:
            await sandbox.files.write(f"{path}/src/pages/{page}.jsx", page_content)
            new_files.append(f"src/pages/{page}.jsx")

    return new_files


# ─────────────────────────────────────────────────────────────
# Node entry point
# ─────────────────────────────────────────────────────────────

async def scaffold_node(state: GraphState, config: RunnableConfig) -> dict:
    """Deterministic scaffolding — install deps, generate App.jsx with routes, set up index.css.

    Zero LLM tokens. Runs in ~10s (mostly npm install).
    On follow-ups: detects new pages from plan and auto-updates App.jsx routes.
    """
    timer = NodeTimer("scaffold")
    configurable = config.get("configurable", {})
    event_queue = configurable.get("event_queue")
    sandbox = configurable.get("sandbox")

    # ── Follow-up path: update routes if new pages, then hand off ──
    if not state.get("is_first_message", True):
        plan = state.get("plan", {})
        new_page_files = []

        if sandbox and plan.get("pages"):
            try:
                new_page_files = await _update_routes_for_new_pages(sandbox, plan, event_queue)
                if new_page_files:
                    print(f"Scaffold follow-up: created {len(new_page_files)} new page stubs")
            except Exception as e:
                print(f"Scaffold follow-up route update failed (non-fatal): {e}")

        safe_send_event(event_queue, {"e": "builder_started", "message": "Generating code for your application..."})
        log_entry = timer.stop(status="skipped (follow-up)" if not new_page_files else "completed (route update)")
        return {
            "scaffold_complete": True,
            "files_created": new_page_files,
            "current_node": "scaffold",
            "execution_log": [log_entry],
        }

    # ── First-build path ──
    try:
        if not sandbox:
            raise Exception("Sandbox not available")

        plan = state.get("plan", {})
        pages = plan.get("pages", ["Home"])
        components = plan.get("components", [])
        dependencies = plan.get("dependencies", [])
        path = "/home/user/react-app"
        files_created = []

        safe_send_event(event_queue, {"e": "builder_started", "message": "Generating code for your application..."})

        # Step 1: Install dependencies (with --legacy-peer-deps fallback for React 19 compat)
        if dependencies:
            dep_str = " ".join(dependencies)
            safe_send_event(event_queue, {
                "e": "tool_started",
                "tool_name": "execute_command",
                "tool_input": {"command": f"npm install {dep_str}"},
            })
            try:
                result = await sandbox.commands.run(f"npm install {dep_str}", cwd=path, timeout=120)
                install_ok = result.exit_code == 0
            except Exception as install_err:
                print(f"Scaffold npm install failed, retrying with --legacy-peer-deps: {str(install_err)[:200]}")
                install_ok = False

            if not install_ok:
                # Retry with --legacy-peer-deps for packages that don't support React 19 yet
                try:
                    result = await sandbox.commands.run(
                        f"npm install --legacy-peer-deps {dep_str}", cwd=path, timeout=120
                    )
                    install_ok = True
                    print(f"Scaffold npm install succeeded with --legacy-peer-deps")
                except Exception as retry_err:
                    print(f"Scaffold npm install failed even with --legacy-peer-deps: {str(retry_err)[:200]}")

            safe_send_event(event_queue, {
                "e": "tool_completed",
                "tool_name": "execute_command",
                "tool_output": f"npm install {dep_str}: {'ok' if install_ok else 'failed'}",
            })

        # Step 2: If plan has only 1 page, make it the root route directly (no Home.jsx)
        # If multiple pages, keep Home as landing page
        if len(pages) == 1 and pages[0] != "Home":
            # Single page app: that page IS the home page at /
            effective_pages = pages
        else:
            # Multi-page: ensure Home is the first page (landing)
            if "Home" not in pages:
                effective_pages = ["Home"] + pages
            else:
                effective_pages = pages

        # Step 3: Generate and write App.jsx
        app_jsx_content = _generate_app_jsx(effective_pages, components)
        await sandbox.files.write(f"{path}/src/App.jsx", app_jsx_content)
        files_created.append("src/App.jsx")
        safe_send_event(event_queue, {"e": "file_created", "message": "Created src/App.jsx"})

        # Step 4: Generate and write index.css
        index_css = '@import "tailwindcss";\n'
        await sandbox.files.write(f"{path}/src/index.css", index_css)
        files_created.append("src/index.css")

        # Step 5: Create placeholder page stubs so imports don't break
        for page in effective_pages:
            if page == "Home" and len(pages) == 1 and pages[0] != "Home":
                # Skip Home — the single page IS the home
                continue
            page_content = f"""import React from 'react'

export default function {page}() {{
  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <h1 className="text-3xl font-bold text-gray-900">{page}</h1>
    </div>
  )
}}
"""
            await sandbox.files.write(f"{path}/src/pages/{page}.jsx", page_content)
            files_created.append(f"src/pages/{page}.jsx")

        print(f"Scaffold complete: {len(files_created)} files, {len(dependencies)} deps")

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
