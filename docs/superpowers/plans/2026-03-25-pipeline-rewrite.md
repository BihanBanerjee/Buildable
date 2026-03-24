# Pipeline Rewrite: Adorable-Style 2-Agent Architecture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Buildable's 6-node LangGraph pipeline with Adorable's simpler 2-agent design (Build agent + Edit agent) while keeping all production infrastructure intact.

**Architecture:** Two independent LangGraph agent loops (agent ⇄ tools, recursion_limit=50). Build agent uses Sonnet 4.5 with `create_app` + `web_search`. Edit agent uses o4-mini with `modify_app` + `chat_message` + `web_search`. Service layer orchestrates sandbox lifecycle, file assembly, and validation.

**Tech Stack:** Python 3.12, FastAPI, LangGraph, langchain-openai (via OpenRouter), E2B sandboxes, Cloudflare R2

**Spec:** `docs/superpowers/specs/2026-03-25-pipeline-rewrite-design.md`

---

## File Structure

### Files to DELETE (11 files)
- `agent/nodes/planner.py`
- `agent/nodes/scaffold.py`
- `agent/nodes/builder.py`
- `agent/nodes/build_checkpoint.py`
- `agent/nodes/fixer.py`
- `agent/nodes/app_start.py`
- `agent/nodes/helpers.py`
- `agent/nodes/__init__.py`
- `agent/graph_builder.py`
- `agent/graph_state.py`
- `agent/formatters.py`

### Files to CREATE (4 files)
- `agent/assembler.py` — Merges generated files with BASE_TEMPLATE
- `agent/sandbox.py` — Sandbox lifecycle (create, check, update, validate)
- `agent/build_agent.py` — Initial build LangGraph agent
- `agent/edit_agent.py` — Edit/follow-up LangGraph agent + error fix

### Files to REWRITE (4 files)
- `agent/tools.py` — Replace all tools with create_app, modify_app, chat_message
- `agent/prompts.py` — Port Adorable's 3 prompts
- `agent/agent.py` — Hardcode models, remove selection logic
- `agent/service.py` — Rewrite orchestration (handle_first_build + handle_follow_up)

### Files to MODIFY (3 files)
- `main.py` — Remove model_choice from ChatPayload, update imports
- `frontend/lib/sse-handlers.ts` — Handle new event format
- `frontend/lib/chat-types.ts` — Simplify BuildStage type
- `frontend/components/chat/BuildProgress.tsx` — Simplify stages

### Files UNCHANGED
- `agent/base_template.py` — Already aligned
- `agent/web_search.py` — Already functional
- `auth/`, `db/`, `utils/`, `alembic/` — No changes

---

### Task 1: Delete Old Pipeline Files

**Files:**
- Delete: `agent/nodes/planner.py`, `agent/nodes/scaffold.py`, `agent/nodes/builder.py`, `agent/nodes/build_checkpoint.py`, `agent/nodes/fixer.py`, `agent/nodes/app_start.py`, `agent/nodes/helpers.py`, `agent/nodes/__init__.py`
- Delete: `agent/graph_builder.py`, `agent/graph_state.py`, `agent/formatters.py`

- [ ] **Step 1: Delete all old pipeline files**

```bash
rm agent/nodes/planner.py agent/nodes/scaffold.py agent/nodes/builder.py
rm agent/nodes/build_checkpoint.py agent/nodes/fixer.py agent/nodes/app_start.py
rm agent/nodes/helpers.py agent/nodes/__init__.py
rm agent/graph_builder.py agent/graph_state.py agent/formatters.py
rmdir agent/nodes
```

- [ ] **Step 2: Commit**

```bash
git add -A agent/nodes/ agent/graph_builder.py agent/graph_state.py agent/formatters.py
git commit -m "chore: delete old 6-node pipeline files"
```

---

### Task 2: Simplify `agent/agent.py` — Hardcoded Models

**Files:**
- Modify: `agent/agent.py`

- [ ] **Step 1: Rewrite agent.py with hardcoded models**

Replace entire file with:

```python
"""
LLM configuration — hardcoded models via OpenRouter (BYOK).

Build agent: anthropic/claude-sonnet-4.5 (high quality initial generation)
Edit agent:  openai/o4-mini (fast follow-up edits)
"""

from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

BUILD_MODEL = "anthropic/claude-sonnet-4.5"
EDIT_MODEL = "openai/o4-mini"


def create_build_llm(api_key: str) -> ChatOpenAI:
    """Create LLM for initial builds (Sonnet 4.5)."""
    return ChatOpenAI(
        model=BUILD_MODEL,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0,
    )


def create_edit_llm(api_key: str) -> ChatOpenAI:
    """Create LLM for edits and error fixes (o4-mini)."""
    return ChatOpenAI(
        model=EDIT_MODEL,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0,
    )
```

- [ ] **Step 2: Commit**

```bash
git add agent/agent.py
git commit -m "feat: hardcode build/edit models (Sonnet 4.5 + o4-mini)"
```

---

### Task 3: Rewrite `agent/prompts.py` — Port Adorable's 3 Prompts

**Files:**
- Modify: `agent/prompts.py`

- [ ] **Step 1: Rewrite prompts.py**

Replace entire file. Port Adorable's prompts from `apps/backend/src/prompt.ts` (lines 3-234), adapting to Python and Buildable's naming. Keep GUARDRAIL_PROMPT and CHAT_RESPONSE_PROMPT as they are (still needed).

Three new prompts to port:
1. `get_build_system_prompt()` — from Adorable's `getSystemPrompt()` (lines 3-119)
2. `get_edit_system_prompt(current_files: dict[str, str])` — from `getEditSystemPrompt()` (lines 122-178)
3. `get_error_fix_prompt(current_files: dict[str, str], build_errors: str)` — from `getErrorFixPrompt()` (lines 180-234)

Key adaptations:
- Replace "Adorable" with "Buildable" in prompt text
- Reference `agent/base_template.py`'s `LOCKED_FILES` for the base files list
- Use Python f-strings instead of JS template literals
- Keep brand colors list, web search workflow, wiring rule, landing page structure

```python
"""
LLM prompts — ported from Adorable's battle-tested prompts.

Three agent prompts:
  1. BUILD_SYSTEM_PROMPT — initial build (create_app + web_search)
  2. EDIT_SYSTEM_PROMPT  — follow-up edits (modify_app + chat_message + web_search)
  3. ERROR_FIX_PROMPT    — error recovery (modify_app only)

Plus guardrail and chat response prompts (unchanged).
"""

from .base_template import BASE_TEMPLATE, LOCKED_FILES


def get_build_system_prompt() -> str:
    """System prompt for the build agent (initial generation)."""
    base_files = "\n- ".join(BASE_TEMPLATE.keys())

    return f"""
You are **Buildable**, an elite AI editor that generates and modifies
production-grade web applications.

==================================================
CRITICAL INSTRUCTION
==================================================
1. **ALWAYS CALL THE TOOL**: You must call "create_app" to generate code.
2. **NO PLACEHOLDERS**: Full, working code only. No "Lorem ipsum", no "Company Name", no fake data.

==================================================
WEB SEARCH — MANDATORY FOR ANY CONTENT PAGE
==================================================
When the user asks for a landing page, company page, product page, or any topic-specific page,
you MUST call web_search BEFORE writing any code.

**SEARCH WORKFLOW — always do this:**

Step 1 — Brand & Identity search:
  Query: "[Company/Topic] official website"
  Extract: official name, tagline/headline, brand colors, logo style, overall tone

Step 2 — Product & Features search:
  Query: "[Company/Topic] features pricing how it works benefits"
  Extract: exact feature names, pricing tiers, value propositions, target audience, testimonials

Step 3 (if needed) — Extra detail search:
  Query: "[Company/Topic] review 2024 OR why use [Company/Topic]"
  Extract: user pain points solved, differentiators, customer quotes

**Query writing rules:**
- Use the EXACT name the user gave, plus descriptive qualifiers
- Never use a vague single-word query — use "Stripe official website" or "Stripe payments API features pricing"
- For ambiguous names: add industry context
- Always aim for queries that return the official site or structured product info

**From the search results, extract ALL of:**
- Real company name & tagline (exact words from their site)
- Hero headline & sub-headline
- Feature names and descriptions (3–6 features)
- Pricing details (if any)
- Customer testimonials or social proof stats (if any)
- CTA copy (e.g. "Start for free", "Book a demo")
- Brand color identity (e.g. "Stripe is known for indigo-purple-blue")

==================================================
COLORS — USE TAILWIND CLASSES, NOT CSS VARIABLES
==================================================
**Do NOT touch src/index.css or the CSS variables.** Leave the base template untouched.

Instead, apply brand-appropriate colors directly using **Tailwind utility classes** in your JSX:
- Pick a primary brand color from the search results (or infer from industry)
- Use Tailwind's full color palette: bg-blue-600, text-indigo-900, bg-emerald-500, etc.
- Apply gradients: bg-gradient-to-r from-blue-600 to-indigo-700
- Use brand colors consistently: hero bg, buttons, highlights, section accents

**Color-by-industry guide (use when brand color is not found in search):**
- Fintech / Payments → indigo-600, blue-700
- Health / Wellness → emerald-500, green-600
- Food / Restaurant → orange-500, amber-600
- E-commerce → violet-600, purple-700
- SaaS / Productivity → slate-800, blue-600
- Creative / Design → pink-500, rose-500
- Real Estate → stone-700, amber-700
- Education → sky-600, cyan-600

**Known brand colors (use these exactly when the company matches):**
- Stripe → from-violet-600 to-indigo-600
- Airbnb → bg-rose-500
- Spotify → bg-green-500
- Notion → bg-black text-white
- Linear → bg-slate-900 text-white with purple accents
- GitHub → bg-gray-900 text-white
- Vercel → bg-black text-white
- Figma → from-purple-500 to-pink-500

==================================================
THE "WIRING" RULE (MOST IMPORTANT)
==================================================
If you create a new component (e.g., LandingPage), you **MUST** also update 'src/App.jsx' to import and render it.
- **NEVER** leave 'src/App.jsx' displaying the default content.
- **ALWAYS** replace the default content of 'src/App.jsx' with your new component.

==================================================
CONTEXT PROVIDERS — CRITICAL
==================================================
If you create a context file (e.g. RecipeContext.jsx), you MUST wrap <Routes> with the Provider in App.jsx.
Example App.jsx structure:
```
import {{ BrowserRouter, Routes, Route }} from 'react-router-dom';
import {{ RecipeProvider }} from './context/RecipeContext';

export default function App() {{
  return (
    <BrowserRouter>
      <RecipeProvider>
        <Routes>
          <Route path="/" element={{<Home />}} />
        </Routes>
      </RecipeProvider>
    </BrowserRouter>
  );
}}
```

==================================================
LANDING PAGE STRUCTURE — always include these sections
==================================================
1. **Navbar** — logo/name, nav links, CTA button (branded color)
2. **Hero** — big headline, sub-headline, primary CTA, secondary CTA or social proof stat
3. **Features** — 3–6 real features with icons (use lucide-react), title, description
4. **Social Proof** — testimonials, logos, or stats (from search results)
5. **Pricing** (if applicable) — real pricing tiers
6. **Footer** — links, copyright

==================================================
FILE SYSTEM RULES
==================================================
1. **Self-Contained**: If you import it, you must create it.
2. **Extensions**: Always use .jsx for components, .js for pure logic. NEVER .ts/.tsx.
3. **Icons**: Use lucide-react imports (e.g., import {{ Home }} from "lucide-react").
4. **Components**: flat in src/components/ (NEVER subdirectories)
5. **Pages**: flat in src/pages/
6. **Pages import with '../'**: import X from '../components/X'
7. **Components import siblings with './'**: import Y from './Y'
8. **export default** for all components and pages

==================================================
TECH STACK
==================================================
- React 18 + Vite + Tailwind CSS v3
- Lucide React (Icons)
- react-router-dom, clsx, tailwind-merge
- JavaScript ONLY (No TypeScript in generated files)

==================================================
BASE FILES (DO NOT MODIFY THESE)
==================================================
- {base_files}

GO.
"""


def get_edit_system_prompt(current_files: dict[str, str]) -> str:
    """System prompt for the edit agent (follow-up modifications)."""
    files_context = "\n\n".join(
        f"=== {path} ===\n{content}"
        for path, content in current_files.items()
    )

    # Use str.format() to avoid f-string double-brace issues with JSX
    template = """
You are **Buildable**, an elite AI editor that modifies existing web applications
based on user requests.

==================================================
CRITICAL INSTRUCTION
==================================================
1. **ALWAYS CALL THE TOOL**: You must call "modify_app" to make changes.
2. **NO PLACEHOLDERS**: Full, working code only.
3. **PRESERVE EXISTING CODE**: Only modify what the user asks for. Keep all other code intact.

==================================================
HOW TO MAKE CHANGES
==================================================
1. Analyze the user's request carefully.
2. Look at the current files below to understand the project structure.
3. Use the "modify_app" tool with the appropriate action:
   - "modify": Change an existing file (provide COMPLETE new content)
   - "create": Add a new file
   - "delete": Remove a file
4. When modifying a file, output the ENTIRE file content with your changes applied.
5. You can modify multiple files in a single tool call.
6. **ALWAYS call "chat_message"** in the same turn as "modify_app" to tell the user what you changed.

==================================================
WEB SEARCH (use when helpful)
==================================================
When the user's request needs current or real-world information, call **web_search** first
with a clear query, then use the results to implement their request accurately.

==================================================
IMPORTANT RULES
==================================================
1. **Complete Files Only**: When modifying, always output the complete file, not just the changed parts.
2. **Maintain Imports**: If you add a new component, update App.jsx to import and use it.
3. **Self-Contained**: If you import something, make sure it exists or create it.
4. **Extensions**: Always use .jsx for React components.

==================================================
TECH STACK
==================================================
- React 18 + Vite + Tailwind CSS v3
- Lucide React (Icons)
- JavaScript ONLY (No TypeScript in generated files)

==================================================
CURRENT PROJECT FILES
==================================================
{files_context}

Now, implement the user's requested changes.
"""
    return template.format(files_context=files_context)


def get_error_fix_prompt(current_files: dict[str, str], build_errors: str) -> str:
    """System prompt for error fix (build validation failures)."""
    files_context = "\n\n".join(
        f"=== {path} ===\n{content}"
        for path, content in current_files.items()
    )

    # Use str.format() to avoid f-string double-brace issues
    template = """
You are **Buildable**, an elite AI editor that fixes build errors in web applications.

==================================================
CRITICAL INSTRUCTION
==================================================
1. **ALWAYS CALL THE TOOL**: You must call "modify_app" to make fixes.
2. **NO PLACEHOLDERS**: Full, working code only.
3. **FIX ALL ERRORS**: Address every build error shown below.

==================================================
BUILD ERRORS TO FIX
==================================================
{build_errors}

==================================================
HOW TO FIX
==================================================
1. Analyze each error message carefully.
2. Look at the relevant files below and understand the issue.
3. Use the "modify_app" tool with action "modify" to fix the broken files.
4. Output the COMPLETE fixed file content.
5. You can fix multiple files in a single tool call.

==================================================
COMMON FIXES
==================================================
- Missing imports: Add the required import statement
- Undefined variables: Define the variable or fix the typo
- Missing components: Create the component or fix the import path
- Syntax errors: Fix the syntax issue

==================================================
TECH STACK
==================================================
- React 18 + Vite + Tailwind CSS v3
- Lucide React (Icons)
- JavaScript ONLY (No TypeScript in generated files)

==================================================
CURRENT PROJECT FILES
==================================================
{files_context}

Now, fix all the build errors and return working code.
"""
    return template.format(files_context=files_context, build_errors=build_errors)


GUARDRAIL_PROMPT = \"\"\"You are an intent classifier for Buildable, a web application builder.

Classify the user's input into exactly one category:

"build" — The user wants to create, modify, fix, or describe a web application, website, UI, dashboard, landing page, game, or any visual/interactive software project. Even vague or single-word descriptions of apps count as "build". Bug reports, error messages, fix requests, and code-related feedback also count as "build".

"chat" — The user is asking a general knowledge question, having casual conversation, asking for help unrelated to building a web app, or requesting something clearly NOT about creating/modifying a web application.

Examples:
- "todo app" → build
- "spotify clone" → build
- "make me a portfolio" → build
- "fix it" → build
- "the import is wrong" → build
- "add dark mode" → build
- "who is PM of India" → chat
- "what is 2+2" → chat
- "hello how are you" → chat

Respond with ONLY the word "build" or "chat". Nothing else.\"\"\"


CHAT_RESPONSE_PROMPT = \"\"\"You are a friendly assistant inside Buildable, an AI-powered web application builder.

The user sent a message that isn't about building a web application. Answer their question helpfully and concisely, then gently remind them that you're here to help build web applications whenever they're ready.

Keep your response under 150 words. Be friendly and natural.\"\"\"
```

**Implementation note:** `get_build_system_prompt()` uses f-strings with `{{` `}}` for literal braces in JSX examples. `get_edit_system_prompt()` and `get_error_fix_prompt()` use `str.format()` to avoid double-brace conflicts — these functions have no JSX examples so `.format()` is clean.

- [ ] **Step 2: Commit**

```bash
git add agent/prompts.py
git commit -m "feat: port Adorable's 3 prompts (build, edit, error fix)"
```

---

### Task 4: Rewrite `agent/tools.py` — New Tool Set

**Files:**
- Modify: `agent/tools.py`

- [ ] **Step 1: Rewrite tools.py**

Replace entire file. Port from Adorable's `modifyTools.ts` and `chatTools.ts`. The key difference from old tools: these tools do NOT write to the sandbox — they just return data for the orchestrator.

```python
"""
Agent tools — pure data tools that do NOT touch the sandbox.

Build agent tools:  create_app, web_search
Edit agent tools:   modify_app, chat_message, web_search
"""

import os
import re
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
```

- [ ] **Step 2: Commit**

```bash
git add agent/tools.py
git commit -m "feat: rewrite tools (create_app, modify_app, chat_message)"
```

---

### Task 5: Create `agent/assembler.py` — Project Assembler

**Files:**
- Create: `agent/assembler.py`

- [ ] **Step 1: Create assembler.py**

Port from Adorable's `projectAssembler.ts`:

```python
"""
Project assembler — merges LLM-generated files with the BASE_TEMPLATE.

The LLM generates only feature files (components, pages, context).
This module overlays them onto the immutable base template to produce
the complete project file set.
"""

import copy
import re

from .base_template import BASE_TEMPLATE


def assemble_project(generated_files: list[dict]) -> dict[str, str]:
    """Merge generated files with BASE_TEMPLATE.

    1. Start with a copy of BASE_TEMPLATE
    2. Overlay generated files
    3. Strip 'import ./App.css' lines (Tailwind-only)
    4. Force vite.config.js from template (never let LLM override)
    5. Return flat {path: content} dict

    Args:
        generated_files: List of {path: str, content: str} from the build agent.

    Returns:
        Complete project as {relative_path: file_content}.
    """
    project = copy.deepcopy(BASE_TEMPLATE)

    for f in generated_files:
        path = f.get("path", "")
        content = f.get("content", "")

        # Strip App.css imports (Tailwind-only system)
        content = re.sub(r"import\s+[\"']\.\/App\.css[\"'];?\n?", "", content)

        project[path] = content

    # Always force the base-template vite.config.js so allowedHosts / host
    # settings required by the E2B sandbox are never lost.
    project["vite.config.js"] = BASE_TEMPLATE["vite.config.js"]

    return project
```

- [ ] **Step 2: Commit**

```bash
git add agent/assembler.py
git commit -m "feat: add project assembler (merge generated files with base template)"
```

---

### Task 6: Create `agent/sandbox.py` — Sandbox Lifecycle

**Files:**
- Create: `agent/sandbox.py`

- [ ] **Step 1: Create sandbox.py**

Port from Adorable's `sandbox.ts`. Use E2B's Python SDK (`e2b_code_interpreter.AsyncSandbox`).

```python
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
            sandbox.commands.run("npm run build 2>&1", cwd=BASE_PATH, timeout=60),
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
        error_msg = f"Build validation error: {str(e)}"
        print(error_msg)
        return {"success": False, "errors": error_msg}
```

- [ ] **Step 2: Commit**

```bash
git add agent/sandbox.py
git commit -m "feat: add sandbox lifecycle (create, check, update, validate)"
```

---

### Task 7: Create `agent/build_agent.py` — Initial Build Agent

**Files:**
- Create: `agent/build_agent.py`

- [ ] **Step 1: Create build_agent.py**

Port from Adorable's `agent.ts`. LangGraph graph with 2 nodes (agent ⇄ tools), conditional edge on tool_calls.

```python
"""
Build agent — LangGraph agent loop for initial project generation.

Graph: agent -> (has tool_calls?) -> tools -> agent -> ... -> END
Tools: create_app, web_search
Model: anthropic/claude-sonnet-4.5 via OpenRouter
"""

import json
from typing import Callable

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesAnnotation, END
from langgraph.prebuilt import ToolNode

from .agent import create_build_llm
from .tools import get_build_tools
from .prompts import get_build_system_prompt


def _safe_json_parse(value: str):
    """Try to parse a JSON string, return None on failure."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _build_graph(api_key: str):
    """Construct the build agent LangGraph."""
    tools = get_build_tools()
    llm = create_build_llm(api_key)
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: MessagesAnnotation.State):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesAnnotation.State):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and (not last.tool_calls or len(last.tool_calls) == 0):
            return "__end__"
        return "tools"

    tool_node = ToolNode(tools)

    workflow = StateGraph(MessagesAnnotation)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge("__start__", "agent")
    workflow.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "__end__": END,
    })
    workflow.add_edge("tools", "agent")

    return workflow.compile()


def _messages_to_files(messages: list) -> list[dict]:
    """Extract generated files from the graph's final messages.

    Scans all ToolMessage results for create_app output.
    Returns list of {path, content} dicts.
    """
    merged = {}  # path -> {path, content} — last write wins

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        content = msg.content
        if isinstance(content, str):
            content = _safe_json_parse(content) or content

        if not isinstance(content, dict):
            continue

        files = content.get("files", [])
        if isinstance(files, list):
            for f in files:
                if isinstance(f, dict) and f.get("path") and f.get("content"):
                    merged[f["path"]] = f

    return list(merged.values())


def _auto_wire_app_jsx(files: list[dict]) -> list[dict]:
    """If no App.jsx was generated but components exist, create a minimal App.jsx.

    Matches Adorable's auto-wiring behavior.
    """
    has_app = any(f["path"] == "src/App.jsx" for f in files)
    if has_app:
        return files

    main_component = next(
        (f for f in files if f["path"].startswith("src/components/")),
        None,
    )
    if not main_component:
        return files

    name = main_component["path"].split("/")[-1].replace(".jsx", "").replace(".js", "")

    files.append({
        "path": "src/App.jsx",
        "content": f'''import React from "react";
import {name} from "./components/{name}";

export default function App() {{
  return (
    <div className="min-h-screen bg-background text-foreground">
      <{name} />
    </div>
  );
}}''',
    })

    return files


async def run_build_stream(
    prompt: str,
    api_key: str,
    on_event: Callable,
) -> dict:
    """Run the build agent with SSE event streaming.

    Args:
        prompt: User's build request.
        api_key: OpenRouter API key.
        on_event: Callback for SSE events.

    Returns:
        {"success": bool, "files": list[dict], "project_name": str} or
        {"success": bool, "error": str}
    """
    system_prompt = get_build_system_prompt()
    initial_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    on_event({"e": "log", "message": "Building your application..."})

    graph = _build_graph(api_key)

    # Stream the graph
    all_messages = list(initial_messages)

    stream = graph.astream(
        {"messages": initial_messages},
        {"recursion_limit": 50},
    )

    async for chunk in stream:
        for node_name, state in chunk.items():
            new_msgs = state.get("messages", [])
            if not isinstance(new_msgs, list):
                new_msgs = [new_msgs]
            all_messages.extend(new_msgs)

            if node_name == "agent":
                on_event({"e": "log", "message": "LLM thinking..."})
            elif node_name == "tools":
                # Check what tool was called
                last_ai = None
                for m in reversed(all_messages):
                    if isinstance(m, AIMessage):
                        last_ai = m
                        break
                tool_name = last_ai.tool_calls[0]["name"] if last_ai and last_ai.tool_calls else "tool"

                if tool_name == "web_search":
                    on_event({"e": "log", "message": "Searching the web..."})
                elif tool_name == "create_app":
                    # Extract file count from tool result
                    last_msg = new_msgs[-1] if new_msgs else None
                    if isinstance(last_msg, ToolMessage):
                        tc = last_msg.content
                        if isinstance(tc, str):
                            tc = _safe_json_parse(tc) or {}
                        if isinstance(tc, dict):
                            file_count = len(tc.get("files", []))
                            if file_count > 0:
                                on_event({"e": "log", "message": f"Generated {file_count} files"})

    # Extract files from messages
    files = _messages_to_files(all_messages)
    files = _auto_wire_app_jsx(files)

    if not files:
        last = all_messages[-1] if all_messages else None
        error = (
            str(last.content)
            if isinstance(last, AIMessage)
            else "Agent failed to generate files."
        )
        return {"success": False, "error": error}

    # Derive project name from first component
    project_name = "Untitled Project"
    main_comp = next(
        (f for f in files if f["path"].startswith("src/components/")),
        None,
    )
    if main_comp:
        raw = main_comp["path"].split("/")[-1].replace(".jsx", "").replace(".js", "")
        import re
        project_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw)

    return {"success": True, "files": files, "project_name": project_name}
```

- [ ] **Step 2: Commit**

```bash
git add agent/build_agent.py
git commit -m "feat: add build agent (LangGraph agent loop with create_app + web_search)"
```

---

### Task 8: Create `agent/edit_agent.py` — Edit Agent + Error Fix

**Files:**
- Create: `agent/edit_agent.py`

- [ ] **Step 1: Create edit_agent.py**

Port from Adorable's `editAgent.ts`. Same LangGraph structure but with modify_app + chat_message + web_search tools.

```python
"""
Edit agent — LangGraph agent loop for follow-up edits and error fixes.

Graph: agent -> (has tool_calls?) -> tools -> agent -> ... -> END
Tools: modify_app, chat_message, web_search
Model: openai/o4-mini via OpenRouter
"""

import json
from typing import Callable

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesAnnotation, END
from langgraph.prebuilt import ToolNode

from .agent import create_edit_llm
from .tools import get_edit_tools
from .prompts import get_edit_system_prompt, get_error_fix_prompt


def _safe_json_parse(value: str):
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _build_edit_graph(api_key: str):
    """Construct the edit agent LangGraph."""
    tools = get_edit_tools()
    llm = create_edit_llm(api_key)
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: MessagesAnnotation.State):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesAnnotation.State):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and (not last.tool_calls or len(last.tool_calls) == 0):
            return "__end__"
        return "tools"

    tool_node = ToolNode(tools)

    workflow = StateGraph(MessagesAnnotation)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge("__start__", "agent")
    workflow.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "__end__": END,
    })
    workflow.add_edge("tools", "agent")

    return workflow.compile()


def _extract_file_changes(messages: list) -> list[dict]:
    """Extract file changes from ToolMessage results."""
    collected = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        content = msg.content
        if isinstance(content, str):
            content = _safe_json_parse(content) or content

        if not isinstance(content, dict):
            continue

        files = content.get("files", [])
        if isinstance(files, list):
            for f in files:
                if isinstance(f, dict) and f.get("path"):
                    collected.append(f)

    return collected


async def run_edit_stream(
    current_files: dict[str, str],
    user_message: str,
    chat_history: list[dict],
    api_key: str,
    on_event: Callable,
) -> list[dict]:
    """Run edit agent with SSE streaming.

    Args:
        current_files: Current project files {path: content}.
        user_message: User's edit request.
        chat_history: Previous conversation messages.
        api_key: OpenRouter API key.
        on_event: SSE callback.

    Returns:
        List of FileChange dicts [{path, content, action}].
    """
    system_prompt = get_edit_system_prompt(current_files)

    messages = [
        {"role": "system", "content": system_prompt},
        *chat_history,
        {"role": "user", "content": user_message},
    ]

    graph = _build_edit_graph(api_key)
    all_messages = list(messages)
    collected_files = []

    try:
        stream = graph.astream({"messages": messages}, {"recursion_limit": 50})

        async for chunk in stream:
            for node_name, state in chunk.items():
                new_msgs = state.get("messages", [])
                if not isinstance(new_msgs, list):
                    new_msgs = [new_msgs]
                all_messages.extend(new_msgs)

                for msg in new_msgs:
                    # Stream AI text tokens
                    if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content:
                        on_event({"e": "token", "content": msg.content})

                    # Process tool results
                    if isinstance(msg, ToolMessage):
                        content = msg.content
                        if isinstance(content, str):
                            content = _safe_json_parse(content) or content

                        if isinstance(content, dict):
                            # modify_app results
                            files = content.get("files", [])
                            for f in files:
                                if isinstance(f, dict) and f.get("path"):
                                    collected_files.append(f)
                                    on_event({"e": "file_update", "file": {"path": f["path"], "action": f.get("action", "modify")}})

                            # chat_message results
                            if content.get("type") == "chat" and content.get("message"):
                                on_event({"e": "token", "content": content["message"]})

        print(f"EditAgent finished. Collected {len(collected_files)} files")
        return collected_files

    except Exception as e:
        print(f"Edit agent error: {e}")
        on_event({"e": "error", "message": f"Failed to process edit: {str(e)}"})
        return []


async def run_error_fix_stream(
    current_files: dict[str, str],
    build_errors: str,
    api_key: str,
    on_event: Callable,
) -> list[dict]:
    """Run error fix with edit agent.

    Args:
        current_files: Current project files.
        build_errors: Build error output.
        api_key: OpenRouter API key.
        on_event: SSE callback.

    Returns:
        List of FileChange dicts.
    """
    system_prompt = get_error_fix_prompt(current_files, build_errors)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Please fix the build errors shown above."},
    ]

    graph = _build_edit_graph(api_key)
    all_messages = list(messages)
    collected_files = []

    try:
        stream = graph.astream({"messages": messages}, {"recursion_limit": 50})

        async for chunk in stream:
            for node_name, state in chunk.items():
                new_msgs = state.get("messages", [])
                if not isinstance(new_msgs, list):
                    new_msgs = [new_msgs]
                all_messages.extend(new_msgs)

                for msg in new_msgs:
                    if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content:
                        on_event({"e": "token", "content": msg.content})

                    if isinstance(msg, ToolMessage):
                        content = msg.content
                        if isinstance(content, str):
                            content = _safe_json_parse(content) or content

                        if isinstance(content, dict):
                            files = content.get("files", [])
                            for f in files:
                                if isinstance(f, dict) and f.get("path"):
                                    collected_files.append(f)
                                    on_event({"e": "file_update", "file": {"path": f["path"], "action": "modify"}})

        print(f"ErrorFix finished. Collected {len(collected_files)} files")
        return collected_files

    except Exception as e:
        print(f"Error fix agent error: {e}")
        on_event({"e": "error", "message": f"Failed to fix errors: {str(e)}"})
        return []
```

- [ ] **Step 2: Commit**

```bash
git add agent/edit_agent.py
git commit -m "feat: add edit agent + error fix (LangGraph with modify_app + chat_message)"
```

---

### Task 9: Rewrite `agent/service.py` — New Orchestration

**Files:**
- Modify: `agent/service.py`

This is the most complex task. The service manages sandbox lifecycle, guardrail classification, and orchestrates the two agents.

- [ ] **Step 1: Rewrite service.py**

Replace entire file. Two main entry points: `handle_first_build()` and `handle_follow_up()`.

Key responsibilities:
- Sandbox management (get/create/reconnect) — keep existing logic from current `Service` class
- Guardrail classification — keep `_classify_prompt()` but use hardcoded model for it
- `handle_first_build()`: guardrail → build agent → assemble_project → create_sandbox → save to R2
- `handle_follow_up()`: guardrail → load files from R2 → edit agent → validation loop (3 attempts) → update sandbox → save to R2

The service should use the new agents, assembler, and sandbox modules. Remove all LangGraph state machine code.

Important changes from current service.py:
1. Remove `self.workflow = get_workflow()` — no more graph compilation
2. Remove `run_agent_stream()` — replace with `handle_first_build()` and `handle_follow_up()`
3. Keep `get_e2b_sandbox()`, `_try_reconnect_sandbox()`, `close_sandbox()`, `_restore_files_from_disk()`, `snapshot_project_files()`, `_save_conversation_history()` — these are still needed
4. New SSE event format: `log`, `token`, `file_update`, `status`, `warning`, `completed`, `error`

```python
"""
Service layer — orchestrates build/edit agents, sandbox lifecycle, and SSE streaming.

Entry points:
  handle_first_build()  — initial project generation
  handle_follow_up()    — follow-up edits with validation loop
"""

import asyncio
import json
import os
import time
import traceback
import uuid
from datetime import datetime
from typing import Dict

from dotenv import load_dotenv
from e2b_code_interpreter import AsyncSandbox
from sqlalchemy import select

from db.base import AsyncSessionLocal
from db.models import Chat, Message
from utils.store import load_json_store, save_json_store

from .assembler import assemble_project
from .build_agent import run_build_stream
from .edit_agent import run_edit_stream, run_error_fix_stream
from .sandbox import (
    create_sandbox,
    update_sandbox_files,
    validate_sandbox_build,
)
from .prompts import GUARDRAIL_PROMPT, CHAT_RESPONSE_PROMPT

load_dotenv()

TEMPLATE_ID = os.getenv("E2B_TEMPLATE_ID", None)
base_path = "/home/user/react-app"


class Service:
    """Orchestrates the 2-agent pipeline with sandbox lifecycle management."""

    def __init__(self) -> None:
        self.sandboxes: Dict[str, AsyncSandbox] = {}
        self.project_timestamps: Dict[str, float] = {}
        self.sandbox_timeout = 1800
        self.storage_base_path = os.path.join(
            os.path.dirname(__file__), "..", "projects"
        )
        os.makedirs(self.storage_base_path, exist_ok=True)

    # ── Sandbox management (kept from previous implementation) ──

    async def get_e2b_sandbox(self, id: str) -> AsyncSandbox:
        """Get or create E2B sandbox for project."""
        current_time = time.time()

        if id in self.sandboxes:
            last_access = self.project_timestamps.get(id, 0)
            if current_time - last_access < self.sandbox_timeout:
                await self.sandboxes[id].set_timeout(self.sandbox_timeout)
                self.project_timestamps[id] = current_time
                return self.sandboxes[id]
            else:
                try:
                    await self.sandboxes[id].kill()
                except Exception:
                    pass
                del self.sandboxes[id]

        sandbox, is_new = await self._try_reconnect_sandbox(id)
        self.sandboxes[id] = sandbox
        await sandbox.set_timeout(self.sandbox_timeout)
        self.project_timestamps[id] = current_time

        if is_new:
            await self._restore_files_from_disk(id, sandbox)

        return sandbox

    async def _try_reconnect_sandbox(self, project_id: str) -> tuple:
        """Reconnect to a previous sandbox or create a fresh one."""
        metadata_file = os.path.join(self.storage_base_path, project_id, "metadata.json")
        sandbox_id = None
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    stored = json.load(f)
                sandbox_id = stored.get("sandbox_id")
            except Exception:
                pass

        if sandbox_id:
            try:
                sandbox = await asyncio.wait_for(
                    AsyncSandbox.reconnect(sandbox_id), timeout=30
                )
                return sandbox, False
            except Exception:
                pass

        if TEMPLATE_ID:
            sandbox = await AsyncSandbox.create(template=TEMPLATE_ID, timeout=1800)
        else:
            sandbox = await AsyncSandbox.create(timeout=1800)
        return sandbox, True

    async def close_sandbox(self, id: str):
        if id in self.sandboxes:
            sandbox = self.sandboxes.pop(id)
            try:
                await sandbox.kill()
            except Exception:
                pass

    async def _restore_files_from_disk(self, project_id: str, sandbox: AsyncSandbox):
        """Restore files from R2 (primary) or local cache to sandbox."""
        from utils.store import load_all_project_files, load_project_metadata

        # Try R2 first
        files_dict = load_all_project_files(project_id)

        if not files_dict:
            # Fallback: load from local cache using metadata
            metadata = load_project_metadata(project_id)
            file_list = metadata.get("files", []) if metadata else []
            if file_list:
                project_dir = os.path.join(self.storage_base_path, project_id)
                for file_path in file_list:
                    local_file = os.path.join(project_dir, file_path.replace("/", "_"))
                    if os.path.exists(local_file):
                        with open(local_file, "r", encoding="utf-8") as f:
                            files_dict[file_path] = f.read()

        if not files_dict:
            print(f"No files to restore for project {project_id}")
            return

        print(f"Restoring {len(files_dict)} files for project {project_id}")

        async def write_one(path: str, content: str):
            try:
                await sandbox.files.write(f"/home/user/react-app/{path}", content)
            except Exception as e:
                print(f"Failed to restore {path}: {e}")

        await asyncio.gather(*[write_one(p, c) for p, c in files_dict.items()])

        try:
            await sandbox.commands.run("rm -rf node_modules/.vite-temp", cwd="/home/user/react-app")
        except Exception:
            pass

    async def snapshot_project_files(self, project_id: str):
        """Snapshot all source files from sandbox to R2 + local cache."""
        if project_id not in self.sandboxes:
            return

        sandbox = self.sandboxes[project_id]

        find_result = await sandbox.commands.run(
            "find src public -type f 2>/dev/null; "
            "test -f package.json && echo package.json; "
            "test -f index.html && echo index.html",
            cwd="/home/user/react-app",
        )
        file_paths = [
            p.strip()
            for p in find_result.stdout.strip().split("\n")
            if p.strip() and not p.startswith(".")
        ]

        if not file_paths:
            return

        files_dict: dict[str, str] = {}

        async def read_file(path: str):
            try:
                content = await sandbox.files.read(f"/home/user/react-app/{path}")
                files_dict[path] = content
            except Exception:
                pass

        await asyncio.gather(*[read_file(p) for p in file_paths])

        if files_dict:
            from utils.store import save_project_files_bulk, save_project_metadata
            save_project_files_bulk(project_id, files_dict)
            save_project_metadata(project_id, list(files_dict.keys()))
            print(f"Snapshotted {len(files_dict)} files for project {project_id}")

    async def _save_conversation_history(self, project_id, user_prompt, success, files_created=None):
        """Save conversation history to context.json."""
        try:
            context = load_json_store(project_id, "context.json")
            history = context.get("conversation_history", [])
            history.append({
                "timestamp": time.time(),
                "user_prompt": user_prompt,
                "success": success,
                "date": datetime.now().isoformat(),
            })
            if len(history) > 10:
                history = history[-10:]
            context["conversation_history"] = history

            if files_created:
                existing = context.get("files_created", [])
                context["files_created"] = list(dict.fromkeys(existing + files_created))

            save_json_store(project_id, "context.json", context)
        except Exception as e:
            print(f"Failed to save conversation history: {e}")

    # ── Guardrail ──

    async def _classify_prompt(self, prompt: str, api_key: str, project_id: str = "", is_first_message: bool = True) -> str:
        """Classify prompt as 'build' or 'chat'."""
        from langchain_core.messages import SystemMessage, HumanMessage
        from .agent import create_edit_llm  # Use edit model (cheap) for classification

        try:
            context_prefix = ""
            if not is_first_message and project_id:
                context = load_json_store(project_id, "context.json")
                files_count = len(context.get("files_created", [])) if context else 0
                if files_count > 0:
                    context_prefix = (
                        f"[Active build session — {files_count} files created. "
                        f"Treat error messages, bug reports, and change requests as \"build\".]\n"
                    )

            llm = create_edit_llm(api_key)
            messages = [
                SystemMessage(content=GUARDRAIL_PROMPT),
                HumanMessage(content=f"{context_prefix}{prompt}"),
            ]
            response = await llm.ainvoke(messages)
            classification = response.content.strip().lower()
            print(f"Prompt classification: '{classification}' for: {prompt[:80]}")
            return "build" if "build" in classification else "chat"
        except Exception as e:
            print(f"Classification failed, defaulting to build: {e}")
            return "build"

    async def _handle_chat_response(self, prompt, project_id, event_queue, api_key):
        """Answer a non-build prompt conversationally."""
        from langchain_core.messages import SystemMessage, HumanMessage
        from .agent import create_edit_llm

        llm = create_edit_llm(api_key)
        messages = [
            SystemMessage(content=CHAT_RESPONSE_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = await llm.ainvoke(messages)
        answer = response.content.strip()

        event_queue.put_nowait({"e": "chat_response", "message": answer})

        async with AsyncSessionLocal() as db:
            msg = Message(
                id=str(uuid.uuid4()),
                chat_id=project_id,
                role="assistant",
                content=answer,
                event_type="chat_response",
            )
            db.add(msg)
            await db.commit()

        event_queue.put_nowait({
            "e": "completed",
            "url": None,
            "success": True,
        })

    # ── Main entry points ──

    async def handle_first_build(self, prompt: str, api_key: str, project_id: str, event_queue: asyncio.Queue):
        """First build flow:
        1. Classify prompt (guardrail)
        2. If chat: respond and return
        3. Run build agent -> get generated files
        4. assemble_project() -> merge with base template
        5. create_sandbox() -> get sandbox_id, url
        6. Save files to R2
        7. Save metadata to DB
        8. Stream events throughout
        """
        try:
            # Parallel: guardrail + sandbox creation (cancel sandbox if chat)
            classification_task = asyncio.create_task(
                self._classify_prompt(prompt, api_key, project_id, is_first_message=True)
            )
            sandbox_task = asyncio.create_task(self.get_e2b_sandbox(project_id))

            classification = await classification_task

            if classification == "chat":
                sandbox_task.cancel()
                try:
                    await sandbox_task
                except asyncio.CancelledError:
                    pass
                await self._handle_chat_response(prompt, project_id, event_queue, api_key)
                return

            # Wait for sandbox (may already be done)
            pre_sandbox = await sandbox_task

            event_queue.put_nowait({"e": "started"})
            event_queue.put_nowait({"e": "log", "message": "Building your application..."})

            workflow_start = time.time()

            # Run build agent
            def on_build_event(event):
                event_queue.put_nowait(event)

            result = await asyncio.wait_for(
                run_build_stream(prompt, api_key, on_build_event),
                timeout=300,  # 5 minute timeout
            )

            if not result["success"]:
                error = result.get("error", "Build failed — no files generated.")
                event_queue.put_nowait({"e": "error", "message": error})
                await self._store_message(project_id, "assistant", error, "error")
                return

            generated_files = result["files"]
            event_queue.put_nowait({"e": "log", "message": f"Created {len(generated_files)} files"})

            # Assemble project (merge with base template)
            project_files = assemble_project(generated_files)

            # Create sandbox
            event_queue.put_nowait({"e": "log", "message": "Setting up sandbox..."})

            def on_sandbox_log(msg):
                event_queue.put_nowait({"e": "log", "message": msg})

            sandbox_result = await create_sandbox(project_files, on_log=on_sandbox_log)

            sandbox = sandbox_result["sandbox"]
            sandbox_id = sandbox_result["sandbox_id"]
            url = sandbox_result["url"]

            # Store sandbox reference
            self.sandboxes[project_id] = sandbox
            self.project_timestamps[project_id] = time.time()

            # Save to DB
            async with AsyncSessionLocal() as db:
                result_db = await db.execute(select(Chat).where(Chat.id == project_id))
                chat = result_db.scalar_one_or_none()
                if chat:
                    chat.app_url = url
                await db.commit()

            # Send completion
            duration = round(time.time() - workflow_start, 2)
            event_queue.put_nowait({
                "e": "completed",
                "success": True,
                "url": url,
                "duration_s": duration,
            })

            await self._store_message(project_id, "assistant", f"Build complete. Preview: {url}", "completed")

            # Background: snapshot + save history
            asyncio.create_task(self._post_build_cleanup(
                project_id, prompt, True,
                [f["path"] for f in generated_files],
                sandbox_id,
            ))

        except asyncio.TimeoutError:
            event_queue.put_nowait({"e": "error", "message": "Build timed out. Please try a simpler prompt."})
        except Exception as e:
            error_msg = str(e)
            print(f"Build error: {error_msg}")
            traceback.print_exc()

            # Detect API errors
            if "402" in error_msg or "credits" in error_msg.lower():
                error_msg = "Insufficient API credits. Please add credits to your API provider."
            elif "429" in error_msg or "rate" in error_msg.lower():
                error_msg = "API rate limit reached. Please wait a moment and try again."
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                error_msg = "API authentication failed. Please check your API key."

            event_queue.put_nowait({"e": "error", "message": error_msg})

    async def handle_follow_up(self, message: str, api_key: str, project_id: str, event_queue: asyncio.Queue):
        """Follow-up edit flow:
        1. Classify prompt (guardrail)
        2. Load current files from R2
        3. Run edit agent -> get FileChange[]
        4. Apply changes to current files
        5. Validation loop (max 3 attempts):
           a. Write temp files to sandbox
           b. validate_sandbox_build()
           c. If fail: run_error_fix_stream() -> apply fixes -> retry
        6. Write final files to live sandbox
        7. Save updated files to R2
        """
        try:
            # Guardrail
            classification = await self._classify_prompt(message, api_key, project_id, is_first_message=False)

            if classification == "chat":
                await self._handle_chat_response(message, project_id, event_queue, api_key)
                return

            event_queue.put_nowait({"e": "started"})

            workflow_start = time.time()

            # Get sandbox
            sandbox = await self.get_e2b_sandbox(project_id)

            # Load current files from R2
            from utils.store import load_all_project_files
            current_files = load_all_project_files(project_id)

            if not current_files:
                # Fallback: read from sandbox
                find_result = await sandbox.commands.run(
                    "find src public -type f 2>/dev/null",
                    cwd="/home/user/react-app",
                )
                file_paths = [p.strip() for p in find_result.stdout.strip().split("\n") if p.strip()]
                current_files = {}
                for path in file_paths:
                    try:
                        content = await sandbox.files.read(f"/home/user/react-app/{path}")
                        current_files[path] = content
                    except Exception:
                        pass

            # Load chat history for context
            chat_history = []
            try:
                async with AsyncSessionLocal() as db:
                    from sqlalchemy import select as sa_select
                    result = await db.execute(
                        sa_select(Message)
                        .where(Message.chat_id == project_id)
                        .order_by(Message.created_at)
                    )
                    messages = result.scalars().all()
                    for msg in messages[-10:]:  # Last 10 messages
                        chat_history.append({"role": msg.role, "content": msg.content})
            except Exception:
                pass

            # Run edit agent
            def on_edit_event(event):
                event_queue.put_nowait(event)

            file_changes = await asyncio.wait_for(
                run_edit_stream(current_files, message, chat_history, api_key, on_edit_event),
                timeout=300,
            )

            if not file_changes:
                event_queue.put_nowait({"e": "error", "message": "Edit agent produced no changes."})
                return

            # Apply changes to current files
            for change in file_changes:
                path = change["path"]
                action = change.get("action", "modify")
                if action == "delete":
                    current_files.pop(path, None)
                else:
                    current_files[path] = change.get("content", "")

            # Validation loop (max 3 attempts)
            max_attempts = 3
            for attempt in range(max_attempts):
                event_queue.put_nowait({"e": "status", "message": f"Validating code{f' (attempt {attempt + 1}/{max_attempts})' if attempt > 0 else ''}..."})

                # Write temp files to sandbox
                await update_sandbox_files(sandbox, current_files)

                # Validate build
                build_result = await validate_sandbox_build(sandbox)

                if build_result["success"]:
                    break

                if attempt < max_attempts - 1:
                    # Run error fix
                    event_queue.put_nowait({"e": "status", "message": f"Fixing errors (attempt {attempt + 1}/{max_attempts})..."})

                    fixes = await run_error_fix_stream(
                        current_files, build_result["errors"], api_key, on_edit_event
                    )

                    # Apply fixes
                    for fix in fixes:
                        path = fix["path"]
                        action = fix.get("action", "modify")
                        if action == "delete":
                            current_files.pop(path, None)
                        else:
                            current_files[path] = fix.get("content", "")
                else:
                    event_queue.put_nowait({"e": "warning", "message": "Max retries reached, saving anyway"})

            # Write final files to sandbox
            await update_sandbox_files(sandbox, current_files)

            # Get URL
            host = sandbox.get_host(port=5173)
            url = f"https://{host}"

            # Completion
            duration = round(time.time() - workflow_start, 2)
            event_queue.put_nowait({
                "e": "completed",
                "success": True,
                "url": url,
                "duration_s": duration,
            })

            await self._store_message(project_id, "assistant", "Edit complete.", "completed")

            # Background cleanup
            asyncio.create_task(self._post_build_cleanup(
                project_id, message, True,
                [c["path"] for c in file_changes],
                None,
            ))

        except asyncio.TimeoutError:
            event_queue.put_nowait({"e": "error", "message": "Edit timed out. Please try a simpler request."})
        except Exception as e:
            error_msg = str(e)
            print(f"Edit error: {error_msg}")
            traceback.print_exc()
            event_queue.put_nowait({"e": "error", "message": error_msg})

    # ── Helpers ──

    async def _store_message(self, chat_id, role, content, event_type):
        try:
            async with AsyncSessionLocal() as db:
                msg = Message(
                    id=str(uuid.uuid4()),
                    chat_id=chat_id,
                    role=role,
                    content=content,
                    event_type=event_type,
                )
                db.add(msg)
                await db.commit()
        except Exception as e:
            print(f"Failed to store message: {e}")

    async def _post_build_cleanup(self, project_id, user_prompt, success, files_created, sandbox_id):
        """Background: snapshot + save history + persist sandbox_id."""
        try:
            await self._save_conversation_history(project_id, user_prompt, success, files_created)
            await self.snapshot_project_files(project_id)

            # Persist sandbox_id for reconnection via metadata.json
            if sandbox_id:
                metadata = load_json_store(project_id, "metadata.json") or {}
                metadata["sandbox_id"] = sandbox_id
                metadata["files"] = files_created or metadata.get("files", [])
                save_json_store(project_id, "metadata.json", metadata)

        except Exception as e:
            print(f"Post-build cleanup error: {e}")


agent_service = Service()
```

- [ ] **Step 2: Commit**

```bash
git add agent/service.py
git commit -m "feat: rewrite service with 2-agent orchestration (build + edit)"
```

---

### Task 10: Update `main.py` — Remove Model Choice, Update Imports

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Update main.py imports and ChatPayload**

Changes needed:
1. Remove `from agent.agent import MODELS, DEFAULT_BUILDER_MODEL` — no longer needed
2. Remove `model_choice` from `ChatPayload`
3. Replace `agent_service.run_agent_stream(...)` calls with `agent_service.handle_first_build(...)` and `agent_service.handle_follow_up(...)`
4. Remove all `builder_model=model_choice` parameters
5. Remove model validation logic from `create_project()`

In `create_project()` (POST /chat):
- Remove: `allowed_models = set(MODELS.values())` and `model_choice` validation
- Remove: `model_choice=model_choice` from `new_chat` creation
- Change: `agent_service.run_agent_stream(...)` → `agent_service.handle_first_build(prompt=prompt, api_key=openrouter_api_key, project_id=chat_id, event_queue=event_queue)`

In `send_message()` (POST /chats/{id}/messages):
- Remove: `chat_model_choice` logic
- Change: `agent_service.run_agent_stream(...)` → `agent_service.handle_follow_up(message=prompt, api_key=openrouter_api_key, project_id=id, event_queue=event_queue)`

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "feat: update routes for 2-agent service (remove model choice)"
```

---

### Task 11: Update Frontend SSE Handlers

**Files:**
- Modify: `frontend/lib/sse-handlers.ts`
- Modify: `frontend/lib/chat-types.ts`
- Modify: `frontend/components/chat/BuildProgress.tsx`

- [ ] **Step 1: Update chat-types.ts**

Simplify `BuildStage` type:

```typescript
export type BuildStage =
  | "building"
  | "validating"
  | "completed";
```

- [ ] **Step 2: Update BuildProgress.tsx**

Simplify stages to just "Building" and "Validating":

```typescript
const STAGES: { key: BuildStage; label: string }[] = [
  { key: "building", label: "Building" },
  { key: "validating", label: "Validating" },
];
```

- [ ] **Step 3: Rewrite sse-handlers.ts**

Handle new event format:
- `log` events → show as progress messages in chat (replaces `thinking`, `progress`, node-transition events)
- `token` events → show as streamed LLM text (for edits)
- `file_update` events → show file activity
- `status` events → show validation progress
- `warning` events → show warnings
- Keep: `started`, `completed`, `error`, `chat_response`, `history`, `cancelled`
- Remove handlers for: `thinking`, `progress`, `tool_started`, `tool_completed`, `builder_started`, `builder_complete`, `code_validator_started`, `code_validator_complete`, `app_check_started`, `app_check_complete`, `planner_started`, `planner_complete`, `enhancer_started`, `description`, `summary`, `file_created`, `file_edited`, `files_created`

New handler additions:

```typescript
// Log events (progress messages during build)
if (data.e === "log") {
  const logMessage = (data.message as string) || "";
  if (!logMessage) return;

  handlers.setMessages((prev) => {
    const lastMsg = prev[prev.length - 1];
    if (lastMsg?.role === "assistant") {
      return [...prev.slice(0, -1), { ...lastMsg, content: logMessage, isProgress: true }];
    }
    return [
      ...prev,
      {
        id: Date.now().toString() + "-log",
        role: "assistant" as const,
        content: logMessage,
        created_at: new Date().toISOString(),
        event_type: "log",
        isProgress: true,
      },
    ];
  });
  return;
}

// Token events (streamed LLM text during edits)
if (data.e === "token") {
  const content = (data.content as string) || "";
  if (!content) return;

  handlers.setMessages((prev) => {
    const lastMsg = prev[prev.length - 1];
    if (lastMsg?.role === "assistant" && !lastMsg.isProgress) {
      return [...prev.slice(0, -1), { ...lastMsg, content: (lastMsg.content || "") + content }];
    }
    return [
      ...prev,
      {
        id: Date.now().toString() + "-token",
        role: "assistant" as const,
        content,
        created_at: new Date().toISOString(),
        event_type: "token",
      },
    ];
  });
  return;
}

// File update events
if (data.e === "file_update") {
  const file = data.file as { path: string; action: string } | undefined;
  if (!file) return;

  const fileMsg = `${file.action === "create" ? "Created" : file.action === "delete" ? "Deleted" : "Modified"} ${file.path}`;
  handlers.setMessages((prev) => {
    const lastMsg = prev[prev.length - 1];
    if (lastMsg?.role === "assistant" && lastMsg.event_type === "file_activity") {
      return [...prev.slice(0, -1), { ...lastMsg, content: (lastMsg.content || "") + "\n" + fileMsg }];
    }
    return [
      ...prev,
      {
        id: Date.now().toString() + "-file",
        role: "assistant" as const,
        content: fileMsg,
        created_at: new Date().toISOString(),
        event_type: "file_activity",
      },
    ];
  });
  return;
}

// Status events (validation progress)
if (data.e === "status") {
  const statusMsg = (data.message as string) || "";
  if (!statusMsg) return;

  handlers.setBuildStage("validating");

  handlers.setMessages((prev) => {
    const lastMsg = prev[prev.length - 1];
    if (lastMsg?.role === "assistant") {
      return [...prev.slice(0, -1), { ...lastMsg, content: statusMsg, isProgress: true }];
    }
    return prev;
  });
  return;
}

// Warning events
if (data.e === "warning") {
  const warnMsg = (data.message as string) || "";
  if (warnMsg) {
    handlers.setMessages((prev) => {
      const lastMsg = prev[prev.length - 1];
      if (lastMsg?.role === "assistant") {
        return [...prev.slice(0, -1), { ...lastMsg, content: `⚠️ ${warnMsg}` }];
      }
      return prev;
    });
  }
  return;
}
```

For the `started` event, set build stage to `"building"`:

```typescript
if (data.e === "started") {
  handlers.setIsSending(false);
  handlers.setIsBuilding(true);
  handlers.setBuildStage("building");
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/lib/sse-handlers.ts frontend/lib/chat-types.ts frontend/components/chat/BuildProgress.tsx
git commit -m "feat: update frontend SSE handlers for new 2-agent event format"
```

---

### Task 12: Integration Smoke Test

- [ ] **Step 1: Verify Python imports**

```bash
cd /Users/bihanbanerjee/Desktop/super30/Buildable
python -c "from agent.service import agent_service; print('Service OK')"
python -c "from agent.build_agent import run_build_stream; print('Build agent OK')"
python -c "from agent.edit_agent import run_edit_stream, run_error_fix_stream; print('Edit agent OK')"
python -c "from agent.assembler import assemble_project; print('Assembler OK')"
python -c "from agent.sandbox import create_sandbox; print('Sandbox OK')"
python -c "from agent.tools import get_build_tools, get_edit_tools; print('Tools OK')"
python -c "from agent.prompts import get_build_system_prompt, get_edit_system_prompt, get_error_fix_prompt; print('Prompts OK')"
```

- [ ] **Step 2: Verify frontend builds**

```bash
cd frontend && npm run build
```

- [ ] **Step 3: Verify backend starts**

```bash
uv run python -c "from main import app; print('App loaded OK')"
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete pipeline rewrite — Adorable-style 2-agent architecture"
```

---

## Summary

| # | Task | Files | Description |
|---|------|-------|-------------|
| 1 | Delete old pipeline | 11 files | Remove all 6-node pipeline code |
| 2 | Simplify agent.py | 1 file | Hardcode Sonnet 4.5 + o4-mini |
| 3 | Rewrite prompts.py | 1 file | Port Adorable's 3 prompts |
| 4 | Rewrite tools.py | 1 file | create_app, modify_app, chat_message |
| 5 | Create assembler.py | 1 file | Merge generated files with base template |
| 6 | Create sandbox.py | 1 file | Sandbox lifecycle management |
| 7 | Create build_agent.py | 1 file | LangGraph build agent loop |
| 8 | Create edit_agent.py | 1 file | LangGraph edit agent + error fix |
| 9 | Rewrite service.py | 1 file | 2-agent orchestration |
| 10 | Update main.py | 1 file | Remove model choice, update calls |
| 11 | Update frontend | 3 files | New SSE event handlers |
| 12 | Smoke test | — | Verify imports + builds |
