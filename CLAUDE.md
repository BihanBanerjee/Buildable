# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Buildable is an AI-powered web application builder that converts natural language descriptions into production-ready React apps. Users describe what they want, a 2-agent LangGraph system generates the code, and an E2B sandbox runs it live. The backend is FastAPI (Python 3.12+, managed with [uv](https://docs.astral.sh/uv/)), the frontend is Next.js 16 with React 19 and Tailwind CSS v4.

## Commands

### Backend
- **Run the app:** `uv run main.py` (starts FastAPI on port 8000)
- **Add a dependency:** `uv add <package>`
- **Sync dependencies:** `uv sync`
- **Run database migrations:** `uv run alembic upgrade head`
- **Create a migration:** `uv run alembic revision --autogenerate -m "description"`
- **Start PostgreSQL:** `docker-compose up -d`

### Frontend
- **Install deps:** `cd frontend && npm install`
- **Dev server:** `cd frontend && npm run dev` (port 3000)
- **Build:** `cd frontend && npm run build`
- **Lint:** `cd frontend && npm run lint`

## Architecture

### Backend (`main.py`, `agent/`, `auth/`, `db/`, `utils/`)

**FastAPI server** (`main.py`) exposes REST + SSE endpoints. Key routes:
- `POST /chat` — create project, start agent, returns SSE stream
- `POST /chats/{id}/messages` — follow-up message to existing project
- `GET /sse/{chat_id}` — SSE endpoint for real-time agent events
- `GET /projects/{id}/files` — list sandbox files
- `GET /projects/{id}/download` — export project as ZIP
- Auth routes under `/auth/` (register, login, refresh, logout, me)

**2-Agent LangGraph system** (`agent/`):

First build: `guardrail → prompt enhancer → build agent → assembler → sandbox`
Follow-up: `guardrail → edit agent → validation loop (max 3) → sandbox`

1. **Guardrail** (`agent/service.py:_classify_prompt`) — classifies every message as "build" or "chat". Uses the edit model (cheap). On follow-ups, injects project context. Distinguishes questions (chat) from change requests (build).
2. **Prompt Enhancer** (`agent/service.py:_enhance_prompt`) — lightweight expansion of vague prompts (first builds only, skipped if prompt >200 chars). Additive only — never alters user specifications.
3. **Build Agent** (`agent/build_agent.py`) — LangGraph agent loop using Sonnet 4.5 via OpenRouter. Tools: `create_app` (generates files), `web_search` (real-time data). Returns pure data — no sandbox I/O.
4. **Edit Agent** (`agent/edit_agent.py`) — LangGraph agent loop using o4-mini. Tools: `modify_app` (edit/create/delete files), `chat_message` (respond to user), `web_search`. Also handles error fixes via `run_error_fix_stream()`.
5. **Assembler** (`agent/assembler.py`) — merges LLM-generated files with base template. Strips conflicting imports, forces locked files from template.
6. **Sandbox** (`agent/sandbox.py`) — E2B sandbox lifecycle: create, write files, install deps, start dev server, validate builds.

Key files:
- `agent/service.py` — orchestration, sandbox lifecycle, SSE event queue, guardrail, prompt enhancer, first build + follow-up entry points
- `agent/agent.py` — LLM configuration (hardcoded models via OpenRouter BYOK)
- `agent/build_agent.py` — LangGraph build agent (Sonnet 4.5, `create_app` + `web_search`)
- `agent/edit_agent.py` — LangGraph edit agent (o4-mini, `modify_app` + `chat_message` + `web_search`) + error fix agent
- `agent/tools.py` — pure-data tools (no sandbox I/O): `create_app`, `modify_app`, `chat_message`, `web_search`
- `agent/prompts.py` — all LLM prompts (guardrail, enhancer, chat response, build, edit, error fix)
- `agent/assembler.py` — merges generated files with base template
- `agent/sandbox.py` — E2B sandbox create/update/validate
- `agent/base_template.py` — base project files (vite.config.js, main.jsx, index.css, etc.) and LOCKED_FILES set

**LOCKED_FILES:** `{vite.config.js, src/main.jsx, index.html, tailwind.config.js, postcss.config.js}` — protected from LLM overwrites at both assembler and service layers. Critical: `vite.config.js` has `allowedHosts: true` for E2B preview.

**Models (hardcoded via OpenRouter BYOK):**
- Build agent: `anthropic/claude-sonnet-4.5` (high quality initial generation)
- Edit agent: `openai/o4-mini` (fast follow-up edits, error fixes, guardrail, prompt enhancement)

**E2B Sandboxes:** Each project gets an isolated Linux sandbox (30-min TTL). Sandbox IDs persist in metadata for reconnection.

**File Storage:** Project snapshots are stored in **Cloudflare R2** (S3-compatible, primary) with local disk as cache. R2 client in `utils/r2.py`, dual-write logic in `utils/store.py`. The VM is stateless — all project data lives in R2.

**Database** (`db/`): PostgreSQL via async SQLAlchemy + asyncpg. Models: User, Chat, Message, RefreshToken. Migrations managed by Alembic (`alembic/versions/`).

**Auth** (`auth/`): JWT-based (HS256). Access tokens + refresh tokens with rotation.

**Validation Loop (edits only):** After edit agent returns changes, the service writes files to sandbox, runs `npm run build`, and if it fails, runs the error-fix agent (o4-mini) to patch errors. Max 3 attempts. First builds skip validation.

**SSE events** streamed to frontend: `started`, `log`, `token`, `file_update`, `status`, `warning`, `completed`, `error`, `chat_response`, `history`, `cancelled`.

### Frontend (`frontend/`)

Next.js App Router with TypeScript. Pages: landing (`/`), auth (`/signin`, `/signup`), chat creation (`/chat`), chat conversation (`/chat/[id]`).

- `api/` — Axios client with auth interceptor, organized by domain (auth.ts, chat.ts)
- `lib/` — SSE event handlers (`sse-handlers.ts`), message types/utils (`chat-types.ts`, `chat-utils.ts`)
- `components/ui/` — shadcn/ui primitives (emerald/slate theme)
- `components/chat/` — chat-specific components (messages, terminal log, preview panel, file viewer with Monaco editor)

**Terminal Log** (`components/chat/TerminalLog.tsx`): Shows real-time build progress with color-coded log lines, progress bar, and elapsed timer. Replaces the old 2-step stepper. Clears immediately on completion.

Design: slate-950 background, emerald-500 accent, Geist + JetBrains Mono fonts.

**Cloudflare Pages Deployment** (`utils/cloudflare.py`): Project names capped at 28 chars to avoid SSL cert issues with hash-prefixed subdomains.

## E2B Sandbox Template

The project uses a custom E2B template (`wk3zdgmi2618nihwhp12`, name: `buildable-react`) with `node_modules` pre-installed to speed up sandbox creation. Config files: `e2b.toml`, `e2b.Dockerfile`, `e2b-package.json`.

**IMPORTANT:** `e2b-package.json` must stay in sync with the `package.json` inside `agent/base_template.py`. When updating base template dependencies:
1. Update `BASE_TEMPLATE["package.json"]` in `agent/base_template.py`
2. Update `e2b-package.json` to match
3. Rebuild the template: `e2b template build --dockerfile e2b.Dockerfile`

If these drift apart, the pre-installed `node_modules` won't match and `npm install` will run a full install, defeating the purpose of the custom template.

## Environment Variables

Required in `.env`: `DATABASE_URL`, `SECRET_KEY`, `E2B_API_KEY`. Optional: `E2B_TEMPLATE_ID`, `R2_ACCESS_KEY`, `R2_SECRET_KEY`, `R2_ENDPOINT`, `R2_BUCKET_NAME`. See `.env` for full list.
