# Pipeline Rewrite: Adorable-Style 2-Agent Architecture

## Overview

Replace the current 6-node LangGraph pipeline (planner -> scaffold -> builder -> build_checkpoint -> fixer -> app_start) with Adorable's simpler 2-agent design. Keep Buildable's production infrastructure (FastAPI, SQLAlchemy, R2, auth, frontend).

## Models

- **Build agent:** `anthropic/claude-sonnet-4.5` via OpenRouter, temperature 0
- **Edit agent:** `openai/o4-mini` via OpenRouter, temperature 0
- **BYOK:** User provides their own OpenRouter API key

## Architecture

### Initial Build Flow

```
User prompt
  -> Build Agent (LangGraph: agent <-> tools, recursion_limit=50)
     Tools: create_app, web_search
  -> assembleProject (merge generated files with BASE_TEMPLATE)
  -> createSandbox (write files, npm install, npm run dev)
  -> Return preview URL
```

### Follow-Up Edit Flow

```
User message + current files
  -> Edit Agent (LangGraph: agent <-> tools, recursion_limit=50)
     Tools: modify_app, chat_message, web_search
  -> Apply file changes
  -> Validation loop (max 3 attempts):
     -> Write temp files to sandbox
     -> npm run build (validate)
     -> If fail: runErrorFixStream -> apply fixes -> retry
  -> Write final files to live sandbox
  -> Return updated preview
```

## Files to DELETE

All current pipeline nodes and orchestration:

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

## Files to CREATE

### `agent/build_agent.py` — Initial Build Agent

LangGraph graph with 2 nodes:
- `agent`: Invokes LLM (Sonnet) with tools bound
- `tools`: ToolNode executes tool calls

Conditional edge: if AIMessage has no tool_calls -> END, else -> tools -> agent

```python
async def run_build_stream(
    prompt: str,
    api_key: str,
    on_event: Callable,  # SSE callback
) -> dict:
    """Run initial build agent. Returns {success, files[], project_name}."""
```

The agent calls `create_app` with all files. create_app does NOT write to sandbox — it just returns the files. The orchestrator handles sandbox creation after the agent finishes.

Web search is called first for landing pages / company pages (prompt instructs this).

### `agent/edit_agent.py` — Edit Agent

Same LangGraph structure (agent <-> tools).

```python
async def run_edit_stream(
    current_files: dict[str, str],
    user_message: str,
    chat_history: list[dict],
    api_key: str,
    on_event: Callable,
) -> list[dict]:
    """Run edit agent. Returns list of FileChange dicts."""

async def run_error_fix_stream(
    current_files: dict[str, str],
    build_errors: str,
    api_key: str,
    on_event: Callable,
) -> list[dict]:
    """Run error fix with edit agent. Returns list of FileChange dicts."""
```

System prompt includes list of all current file paths so the LLM knows what exists.

### `agent/assembler.py` — Project Assembler

```python
def assemble_project(generated_files: list[dict]) -> dict[str, str]:
    """Merge generated files with BASE_TEMPLATE.

    1. Start with copy of BASE_TEMPLATE
    2. Overlay generated files
    3. Strip 'import ./App.css' lines (Tailwind-only)
    4. Force vite.config.js from template (never let LLM override)
    5. Return flat {path: content} dict
    """
```

### `agent/sandbox.py` — Sandbox Lifecycle

```python
async def create_sandbox(
    project_files: dict[str, str],
    on_log: Callable | None = None,
) -> dict:
    """Create E2B sandbox, write files, install deps, start Vite.
    Returns {sandbox_id, url}.
    Sandbox timeout: 3600s (1 hour).
    """

async def is_sandbox_alive(sandbox_id: str) -> bool:
    """Check if sandbox is alive (echo test, 5s timeout)."""

async def update_sandbox_files(
    sandbox_id: str,
    files: dict[str, str],
) -> None:
    """Write files to sandbox. Restart Vite if not running."""

async def validate_sandbox_build(sandbox_id: str) -> dict:
    """Run 'npm run build 2>&1' with 60s timeout.
    Returns {success: bool, errors: str | None}.
    """
```

## Files to MODIFY

### `agent/tools.py` — Complete Rewrite

Replace all current tools with:

```python
# 1. create_app(files: list[{path, content}]) -> str
#    Validates paths, returns normalized file list
#    Does NOT write to sandbox

# 2. modify_app(files: list[{path, content, action}]) -> str
#    action: "create" | "modify" | "delete"
#    Returns applied changes
#    Does NOT write to sandbox

# 3. chat_message(message: str) -> str
#    Returns the message for user display

# 4. web_search — re-export from agent/web_search.py
```

### `agent/prompts.py` — Replace All Prompts

Port Adorable's 3 prompts:

1. **BUILD_SYSTEM_PROMPT** — Initial build instructions
   - ALWAYS call create_app with complete code
   - Web search mandatory for landing/content pages
   - Use Tailwind classes for colors
   - Wiring rule: always update App.jsx when creating components
   - Base files are locked
   - Tech stack: React 18 + Vite + Tailwind v3 + Lucide React
   - Known brand colors by industry

2. **EDIT_SYSTEM_PROMPT** — Edit mode instructions (takes current_files as param)
   - Full context of current files included
   - Use modify_app only
   - Preserve existing code unless asked to change
   - Output COMPLETE file content for modifications
   - Use chat_message to confirm changes

3. **ERROR_FIX_PROMPT** — Error recovery (takes current_files + build_errors)
   - Build errors injected
   - Use modify_app action "modify"
   - Output complete fixed files
   - Common fixes list

### `agent/agent.py` — Simplify LLM Creation

```python
BUILD_MODEL = "anthropic/claude-sonnet-4.5"
EDIT_MODEL = "openai/o4-mini"

def create_build_llm(api_key: str):
    return ChatOpenAI(model=BUILD_MODEL, api_key=api_key, temperature=0,
                      openai_api_base="https://openrouter.ai/api/v1")

def create_edit_llm(api_key: str):
    return ChatOpenAI(model=EDIT_MODEL, api_key=api_key, temperature=0,
                      openai_api_base="https://openrouter.ai/api/v1")
```

Remove: `get_fast_model()`, model selection logic, `max_tokens` param.

### `agent/service.py` — Rewrite Orchestration

Two main functions:

```python
async def handle_first_build(prompt, api_key, project_id, event_queue):
    """First build flow:
    1. Classify prompt (guardrail: build vs chat)
    2. If chat: respond with chat_response, return
    3. Run build agent -> get generated files
    4. assemble_project(files) -> merge with base template
    5. create_sandbox(project_files) -> get sandbox_id, url
    6. Save files to R2
    7. Save metadata to DB
    8. Stream events throughout
    """

async def handle_follow_up(message, api_key, project_id, event_queue):
    """Follow-up edit flow:
    1. Load current files from R2
    2. Extract sandbox_id from stored metadata
    3. Run edit agent -> get FileChange[]
    4. Apply changes to current files
    5. Validation loop (max 3 attempts):
       a. update_sandbox_files (temporary)
       b. validate_sandbox_build()
       c. If fail: run_error_fix_stream() -> apply fixes -> retry
    6. Write final files to live sandbox (restart Vite if needed)
    7. Save updated files to R2
    8. Stream events throughout
    """
```

Remove: All LangGraph state machine code, node dispatching, graph compilation.

### `agent/base_template.py` — No Changes

Already aligned with Adorable (Tailwind v3, HSL tokens, PostCSS).

### `agent/web_search.py` — No Changes

Already functional, used by both agents.

## SSE Event Format

### Initial Build Events

```python
{"e": "started"}
{"e": "log", "message": "Building your application..."}
{"e": "log", "message": "Searching the web..."}           # if web_search called
{"e": "log", "message": "Generating code..."}              # create_app called
{"e": "log", "message": "Created 12 files"}
{"e": "log", "message": "Setting up sandbox..."}
{"e": "log", "message": "Installing dependencies..."}
{"e": "log", "message": "Starting dev server..."}
{"e": "completed", "success": true, "url": "https://..."}
```

### Follow-Up Edit Events

```python
{"e": "started"}
{"e": "token", "content": "..."}                           # LLM streamed thinking
{"e": "file_update", "file": {"path": "...", "action": "modify"}}
{"e": "status", "message": "Validating code..."}
{"e": "status", "message": "Fixing errors (attempt 1/3)..."}
{"e": "warning", "message": "Max retries reached, saving anyway"}
{"e": "completed", "success": true, "url": "https://..."}
```

### Error Events

```python
{"e": "error", "message": "Build failed: ..."}
```

## Frontend Changes

### `frontend/lib/sse-handlers.ts`

Update to handle new event format:
- `log` events → show as progress messages in chat
- `token` events → show as streamed LLM text
- `file_update` events → show file activity
- `status` events → show validation progress
- Remove handlers for: `thinking`, `progress`, `tool_started`, `tool_completed`, `builder_started`, `builder_complete`, `code_validator_started`, `code_validator_complete`, `app_check_started`, `app_check_complete`, `planner_started`, `planner_complete`, `enhancer_started`, `description`, `summary`

### `frontend/components/chat/BuildProgress.tsx`

Simplify build stages. Remove: Understanding, Planning, Building, Validating, Testing.
Replace with simpler: Building → Validating (only shown during edits).

## DB/API Changes

### `main.py` Routes

No route changes needed. The `/chat` and `/chat/{id}/message` endpoints call into `service.py` which we're rewriting internally.

### Graph State

Remove `agent/graph_state.py` entirely. No more TypedDict state machine.
Service.py manages state as local variables within each handler function.

## Migration Path

1. Delete all pipeline nodes and graph files
2. Create new agent files (build_agent, edit_agent, assembler, sandbox)
3. Rewrite tools.py, prompts.py, agent.py, service.py
4. Update frontend SSE handlers
5. Test initial build
6. Test follow-up edits
7. Test error fix loop

## What Stays Unchanged

- `auth/` — JWT auth, routes, middleware
- `db/` — Models, migrations, session management
- `utils/store.py` — R2 + local file storage
- `utils/r2.py` — Cloudflare R2 client
- `utils/cloudflare.py` — Cloudflare Pages deployment
- `alembic/` — Database migrations
- `frontend/` — Mostly unchanged (minor SSE handler updates)
- `agent/base_template.py` — Already aligned
- `agent/web_search.py` — Already functional
