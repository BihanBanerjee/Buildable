# Buildable

**Describe it. Build it. Ship it.**

Buildable is an AI-powered web application builder that converts natural language descriptions into production-ready React apps. A multi-agent LangGraph pipeline plans, scaffolds, and generates code in an isolated E2B sandbox — with live preview, iterative chat refinement, and one-click export.

## How It Works

1. **Describe** — Write what you want in plain English
2. **Build** — A 6-node AI pipeline plans, scaffolds, generates, validates, and fixes your app
3. **Iterate** — Refine with follow-up messages; the builder makes surgical edits, not full rewrites
4. **Ship** — Preview live, download as ZIP, deploy anywhere

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
│  │ planner → scaffold → builder → checkpoint ⇄ fixer  │ │
│  │                                         → app_start│ │
│  └────────────────────────────────────────────────────┘ │
│  ┌─── Follow-up ──────────────────────────────────────┐ │
│  │ guardrail → builder → checkpoint ⇄ fixer           │ │
│  │                                  → app_start       │ │
│  └────────────────────────────────────────────────────┘ │
│                                                         │
│  Auth (JWT) · PostgreSQL · Cloudflare R2 · E2B Sandbox  │
└─────────────────────────────────────────────────────────┘
```

### Pipeline Nodes

| Node | Role | LLM? |
|------|------|------|
| **Planner** | Enhances prompt, generates structured plan (components, pages, deps) | Fast model |
| **Scaffold** | Installs deps, generates routes + page stubs | No |
| **Builder** | Generates code via tool calls (`create_file`, `edit_file`) | User-selected |
| **Checkpoint** | Runs `vite build`, detects missing packages | No |
| **Fixer** | Fixes build/runtime errors (max 2 retries) | Gemini Flash |
| **App Start** | Starts Vite dev server, checks HTTP 200 + runtime errors | No |

Follow-ups skip planner/scaffold and use `edit_file` for surgical changes instead of full rewrites.

## Tech Stack

**Backend:** Python 3.12+, FastAPI, LangGraph, LangChain, SQLAlchemy (async), E2B, Cloudflare R2, Alembic

**Frontend:** Next.js 16 (App Router), TypeScript, Tailwind CSS v4, shadcn/ui, Monaco Editor, Axios, SSE

**Infrastructure:** DigitalOcean, Nginx, Certbot, Terraform, Neon PostgreSQL, Vercel

**Models:** User-selected via OpenRouter (BYOK) — Gemini 2.5 Pro or Claude Sonnet 4

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
| `FRONTEND_URL` | No | Frontend URL for CORS (default: `http://localhost:3000`) |

## Project Structure

```
├── main.py                    # FastAPI entry point
├── agent/
│   ├── service.py             # Orchestration + sandbox lifecycle
│   ├── graph_builder.py       # LangGraph state machine
│   ├── tools.py               # Sandbox tools (file CRUD, exec)
│   ├── prompts.py             # LLM prompts
│   └── nodes/                 # Pipeline nodes (planner → app_start)
├── auth/                      # JWT auth (register, login, refresh)
├── db/                        # SQLAlchemy models + base
├── utils/
│   ├── r2.py                  # Cloudflare R2 client
│   └── store.py               # Dual-write (R2 + local cache)
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
