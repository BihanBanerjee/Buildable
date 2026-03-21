# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Buildable is an AI-powered web application builder that converts natural language descriptions into production-ready React apps. Users describe what they want, a multi-agent LangGraph system plans and builds the code, and an E2B sandbox runs it live. The backend is FastAPI (Python 3.12+, managed with [uv](https://docs.astral.sh/uv/)), the frontend is Next.js 16 with React 19 and Tailwind CSS v4.

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
- `POST /chat/{id}/message` — follow-up message to existing project
- `GET /chat/stream/{chat_id}` — SSE endpoint for real-time agent events
- `GET /projects/{id}/files` — list sandbox files
- `GET /projects/{id}/download` — export project as ZIP
- Auth routes under `/auth/` (register, login, refresh, logout, me)

**LangGraph multi-agent system** (`agent/`) — 6-node pipeline:

First build: `planner → scaffold → builder → build_checkpoint ⇄ fixer → app_start`
Follow-up: `guardrail → builder → build_checkpoint ⇄ fixer → app_start` (planner/scaffold skipped)

1. **Planner** (`agent/nodes/planner.py`) — enhances vague prompts + generates structured plan (components, pages, deps, file structure). Skipped on follow-ups.
2. **Scaffold** (`agent/nodes/scaffold.py`) — deterministic (no LLM): installs deps, generates App.jsx with routes, index.css, page stubs. Skipped on follow-ups.
3. **Builder** (`agent/nodes/builder.py`) — generates code via LLM tool calls. First build uses `create_file` + `write_multiple_files`. Follow-ups use `read_file` + `edit_file` + `create_file` for surgical edits.
4. **Build Checkpoint** (`agent/nodes/build_checkpoint.py`) — runs `vite build` to check for compile errors, detects missing packages.
5. **Fixer** (`agent/nodes/fixer.py`) — always uses Gemini Flash regardless of builder model. Fixes build/runtime errors with `read_file`, `create_file`, `execute_command` (npm install only).
6. **App Start** (`agent/nodes/app_start.py`) — starts Vite dev server, checks HTTP 200, detects runtime errors in Vite logs. Routes to fixer if errors found.

Key files:
- `agent/service.py` — orchestration, sandbox lifecycle, SSE event queue, guardrail classification
- `agent/graph_builder.py` — constructs the LangGraph state machine with conditional edges
- `agent/graph_state.py` — TypedDict state schema
- `agent/nodes/` — individual node implementations (planner, scaffold, builder, build_checkpoint, fixer, app_start)
- `agent/nodes/helpers.py` — shared utilities: NodeTimer, progress ticker, stream_agent_events
- `agent/tools.py` — sandbox tools (file CRUD, shell exec, edit_file, context save/load)
- `agent/prompts.py` — all LLM system/user prompts

**E2B Sandboxes:** Each project gets an isolated Linux sandbox (30-min TTL). Sandbox IDs persist in metadata for reconnection.

**File Storage:** Project snapshots are stored in **Cloudflare R2** (S3-compatible, primary) with local disk as cache. R2 client in `utils/r2.py`, dual-write logic in `utils/store.py`. The VM is stateless — all project data lives in R2.

**Database** (`db/`): PostgreSQL via async SQLAlchemy + asyncpg. Models: User, Chat, Message, RefreshToken. Migrations managed by Alembic (`alembic/versions/`).

**Auth** (`auth/`): JWT-based (HS256). Access tokens + refresh tokens with rotation.

**Guardrail:** Runs on every message (first + follow-ups). Classifies prompts as "build" or "chat". On follow-ups, injects project context so error messages are correctly classified as build requests. On first message, runs in parallel with sandbox creation; cancels sandbox if classified as chat.

**SSE events** streamed to frontend: `thinking`, `tool_started`, `tool_completed`, `file_created`, `file_edited`, `validation_error`, `runtime_error`, `progress`, `token_update`, etc.

### Frontend (`frontend/`)

Next.js App Router with TypeScript. Pages: landing (`/`), auth (`/signin`, `/signup`), chat creation (`/chat`), chat conversation (`/chat/[id]`).

- `api/` — Axios client with auth interceptor, organized by domain (auth.ts, chat.ts)
- `lib/` — SSE event handlers (`sse-handlers.ts`), message types/utils (`chat-types.ts`, `chat-utils.ts`)
- `components/ui/` — shadcn/ui primitives (emerald/slate theme)
- `components/chat/` — chat-specific components (messages, preview panel, file viewer with Monaco editor)

Design: slate-950 background, emerald-500 accent, Geist + JetBrains Mono fonts.

### Model Selection

Users choose LLM per project via OpenRouter (BYOK): Gemini 2.5 Pro or Claude Sonnet 4. Planner uses the fast variant (Flash/Haiku). Fixer always uses Gemini Flash. Configured via `langchain-openai` provider pointed at OpenRouter.

## Environment Variables

Required in `.env`: `DATABASE_URL`, `SECRET_KEY`, `E2B_API_KEY`. Optional: `E2B_TEMPLATE_ID`, `R2_ACCESS_KEY`, `R2_SECRET_KEY`, `R2_ENDPOINT`, `R2_BUCKET_NAME`. See `.env` for full list.
