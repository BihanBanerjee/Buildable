# Buildable

**Describe it. Build it. Ship it.**

Buildable is an AI-powered web application builder that converts natural language descriptions into production-ready React apps. A 2-agent LangGraph system generates and iterates on code in an isolated E2B sandbox — with live preview, iterative chat refinement, and one-click deployment to Cloudflare Pages.

## How It Works

1. **Describe** — Write what you want in plain English
2. **Build** — A build agent (Sonnet 4.5) generates your app with automatic prompt enhancement
3. **Iterate** — Refine with follow-up messages; an edit agent (o4-mini) makes surgical changes with build validation
4. **Ship** — Preview live, download as ZIP, or deploy to Cloudflare Pages

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Frontend                           │
│          Next.js 16 · TypeScript · Tailwind v4          │
│          shadcn/ui · Monaco Editor · SSE Client         │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTPS (SSE)
┌──────────────────────▼──────────────────────────────────┐
│                    Backend (FastAPI)                     │
│                                                         │
│  ┌─── First Build ────────────────────────────────────┐ │
│  │ guardrail → enhancer → build agent → assembler     │ │
│  │                                    → sandbox       │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌─── Follow-up ──────────────────────────────────────┐ │
│  │ guardrail → edit agent → validate ⇄ error fix      │ │
│  │                        → sandbox                   │ │
│  └────────────────────────────────────────────────────┘ │
│                                                         │
│  Auth (JWT) · PostgreSQL · Cloudflare R2 · E2B Sandbox  │
└─────────────────────────────────────────────────────────┘
```

### 2-Agent System

| Agent | Model | Tools | Role |
|-------|-------|-------|------|
| **Build** | Sonnet 4.5 | `create_app`, `web_search` | Initial app generation from prompt |
| **Edit** | o4-mini | `modify_app`, `chat_message`, `web_search` | Follow-up edits + error fixes |

**Supporting steps:**
- **Guardrail** — Classifies every message as "build" or "chat" (questions vs change requests)
- **Prompt Enhancer** — Lightly expands vague prompts (e.g. "todo app" → adds key features). Additive only, skipped if prompt is already detailed
- **Assembler** — Merges LLM files with base template, protects locked config files
- **Validation Loop** — On edits: write → `npm run build` → if fail → error fix agent → retry (max 3)

## Tech Stack

**Backend:** Python 3.12+, FastAPI, LangGraph, LangChain, SQLAlchemy (async), E2B, Cloudflare R2, Alembic

**Frontend:** Next.js 16 (App Router), TypeScript, Tailwind CSS v4, shadcn/ui, Monaco Editor, Axios, SSE

**Infrastructure:** DigitalOcean, Nginx, Certbot, Terraform, Neon PostgreSQL, Vercel

**Models:** Hardcoded via OpenRouter (BYOK) — Sonnet 4.5 for builds, o4-mini for edits/guardrail/enhancement

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Node.js 24+
- Docker (for PostgreSQL)
- [E2B API key](https://e2b.dev)

### Setup

```bash
# Clone
git clone https://github.com/BihanBanerjee/Buildable.git
cd Buildable

# Backend
cp .env.example .env          # Fill in required values
docker-compose up -d           # Start PostgreSQL
uv sync                        # Install Python deps
uv run alembic upgrade head    # Run migrations
uv run main.py                 # Start backend on :8000

# Frontend
cd frontend
npm install
npm run dev                    # Start frontend on :3000
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string (asyncpg) |
| `SECRET_KEY` | Yes | JWT signing key |
| `E2B_API_KEY` | Yes | E2B sandbox API key |
| `E2B_TEMPLATE_ID` | No | Custom E2B template |
| `R2_ACCESS_KEY` | No | Cloudflare R2 access key |
| `R2_SECRET_KEY` | No | Cloudflare R2 secret key |
| `R2_ENDPOINT` | No | Cloudflare R2 endpoint URL |
| `R2_BUCKET_NAME` | No | R2 bucket name |
| `CLOUDFLARE_ACCOUNT_ID` | No | Cloudflare account ID (for deployment) |
| `CLOUDFLARE_API_TOKEN` | No | Cloudflare API token (for deployment) |
| `FRONTEND_URL` | No | Frontend URL for CORS (default: `http://localhost:3000`) |

## Project Structure

```
├── main.py                    # FastAPI entry point
├── agent/
│   ├── service.py             # Orchestration + sandbox lifecycle
│   ├── agent.py               # LLM configuration (models, OpenRouter)
│   ├── build_agent.py         # Build agent (Sonnet 4.5)
│   ├── edit_agent.py          # Edit agent + error fix (o4-mini)
│   ├── tools.py               # Pure-data tools (create_app, modify_app, etc.)
│   ├── prompts.py             # All LLM prompts
│   ├── assembler.py           # Merge generated files with base template
│   ├── sandbox.py             # E2B sandbox lifecycle
│   └── base_template.py       # Base project files + LOCKED_FILES
├── auth/                      # JWT auth (register, login, refresh)
├── db/                        # SQLAlchemy models + base
├── utils/
│   ├── r2.py                  # Cloudflare R2 client
│   ├── store.py               # Dual-write (R2 + local cache)
│   └── cloudflare.py          # Cloudflare Pages deployment
├── alembic/                   # Database migrations
├── frontend/
│   ├── app/                   # Next.js pages (/, /chat, /chat/[id])
│   ├── components/            # UI components (shadcn + chat)
│   ├── lib/                   # SSE handlers, types, utils
│   └── api/                   # Axios client (auth, chat)
├── deploy/                    # Terraform + systemd service
├── nginx/                     # Nginx config + SSL setup
└── .github/workflows/         # CI/CD pipelines
```

## CI/CD

- **CI** — Runs on every push: frontend lint + build, backend ruff + syntax check
- **CD** — Auto-deploys backend to DigitalOcean after CI passes
- **Frontend** — Auto-deployed by Vercel on push

## License

MIT
