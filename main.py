from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import asyncio
import json
import os
import io
import zipfile
import traceback
import uuid
from fastapi import Depends

from sqlalchemy import select, delete
from agent.service import agent_service
from auth.router import router
from auth.schema import ProjectsListResponse
from db.models import User, Chat, Message
from auth.dependencies import get_current_user
from sqlalchemy.ext.asyncio import AsyncSession
from db.base import get_db, AsyncSessionLocal

from auth.utils import decode_token
from utils.crypto import decrypt_api_key
from agent.agent import MODELS, DEFAULT_BUILDER_MODEL


app = FastAPI(title="Buildable")

origins = [
    "http://localhost:3000",
    # "https://webbuilder.elevenai.xyz",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router=router)

active_streams: dict[str, asyncio.Queue] = {}
active_runs: dict[str, asyncio.Task] = {}


class ChatPayload(BaseModel):
    prompt: str
    model_choice: str = "google/gemini-2.5-pro"  # Full OpenRouter model ID


class ChatMessagePayload(BaseModel):
    prompt: str


class ProjectFilesResponse(BaseModel):
    files: list[str]
    sandbox_active: bool


@app.get("/")
async def get_health():
    return {"message": "Welome", "status": "Healthy"}


@app.get("/chats/{id}/messages")
async def get_chat_messages(
    id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get message history for a chat"""
    # Verify the chat exists and belongs to the user
    result = await db.execute(select(Chat).where(Chat.id == id))
    chat = result.scalar_one_or_none()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    if chat.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this chat")

    # Get all messages for the chat
    result = await db.execute(
        select(Message)
        .where(Message.chat_id == id)
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()

    return {
        "chat": {
            "id": chat.id,
            "title": chat.title,
            "app_url": chat.app_url,
            "created_at": chat.created_at
        },
        "messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "event_type": msg.event_type,
                "created_at": msg.created_at
            }
            for msg in messages
        ]
    }


@app.post("/chat")
async def create_project(
    payload: ChatPayload,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Generate UUID on backend
    chat_id = str(uuid.uuid4())

    prompt = payload.prompt

    if not prompt:
        return JSONResponse({"error": "Too short or no description"}, status_code=400)

    # Decrypt user's OpenRouter API key
    if not current_user.encrypted_openrouter_key:
        return JSONResponse(
            {"error": "No API key", "message": "Please add your OpenRouter API key in Settings before building."},
            status_code=400,
        )
    openrouter_api_key = decrypt_api_key(current_user.encrypted_openrouter_key)

    if chat_id in active_runs:
        return JSONResponse(
            {"error": "Project is being created. Kindly wait"}, status_code=400
        )

    allowed_models = set(MODELS.values())
    model_choice = payload.model_choice if payload.model_choice in allowed_models else DEFAULT_BUILDER_MODEL

    new_chat = Chat(
        id=chat_id,
        user_id=current_user.id,
        title=prompt[:100] if len(prompt) > 100 else prompt,
        model_choice=model_choice,
    )

    db.add(new_chat)
    await db.commit()

    # Create initial user message
    user_message = Message(
        id=str(uuid.uuid4()),
        chat_id=chat_id,
        role="user",
        content=prompt
    )
    db.add(user_message)
    await db.commit()

    # Get or create event queue for this chat
    event_queue = active_streams.get(chat_id)
    if not event_queue:
        event_queue = asyncio.Queue()
        active_streams[chat_id] = event_queue

    # Start agent task in background
    async def agent_task():
        try:
            await agent_service.run_agent_stream(
                prompt=prompt, id=chat_id, event_queue=event_queue,
                openrouter_api_key=openrouter_api_key, builder_model=model_choice,
                is_first_message=True,
            )
        except Exception as e:
            print(f"Agent error: {e}")
            traceback.print_exc()

            try:
                async with AsyncSessionLocal() as error_db:
                    error_message = Message(
                        id=str(uuid.uuid4()),
                        chat_id=chat_id,
                        role="assistant",
                        content=f"Build failed: {str(e)}",
                        event_type="error"
                    )
                    error_db.add(error_message)
                    await error_db.commit()
            except Exception as db_err:
                print(f"Failed to store error message: {db_err}")

            try:
                event_queue.put_nowait({
                    "e": "error",
                    "message": f"Build failed: {str(e)}"
                })
            except Exception as queue_err:
                print(f"Failed to send error to event queue: {queue_err}")
        finally:
            if chat_id in active_runs:
                del active_runs[chat_id]

    # Store the task
    active_runs[chat_id] = asyncio.create_task(agent_task())

    return {
        "status": "success",
        "message": "Chat created and agent started.",
        "chat_id": chat_id,
    }


_LIST_FILES_SCRIPT = """
import os
import json

def should_exclude(path):
    exclude_dirs = ['node_modules', '.git', '__pycache__', '.next', 'dist', 'build', '.venv', 'venv']
    exclude_files = ['.DS_Store', 'package-lock.json', 'yarn.lock']
    parts = path.split(os.sep)
    for part in parts:
        if part in exclude_dirs:
            return True
    if os.path.basename(path) in exclude_files:
        return True
    return False

def list_files_recursive(path):
    file_structure = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', '__pycache__', '.next', 'dist', 'build', '.venv', 'venv']]
        for name in files:
            relative_path = os.path.relpath(os.path.join(root, name), path)
            if not should_exclude(relative_path):
                file_structure.append(relative_path)
    return file_structure

react_app_path = "/home/user/react-app"
if os.path.exists(react_app_path):
    print(json.dumps(list_files_recursive(react_app_path)))
else:
    print(json.dumps([]))
"""


async def _list_sandbox_files(sandbox) -> list:
    """List all project files in the sandbox, excluding build artifacts."""
    await sandbox.files.write("/tmp/list_files.py", _LIST_FILES_SCRIPT)
    proc = await sandbox.commands.run("python /tmp/list_files.py", cwd="/tmp")
    if proc.exit_code != 0:
        raise Exception(f"Failed to list files: {proc.stderr}")
    return json.loads(proc.stdout)


@app.get("/projects/{id}/files", response_model=ProjectFilesResponse)
async def get_project_files(id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Chat).where(Chat.id == id))
    chat = result.scalar_one_or_none()
    if not chat or chat.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")

    # Try live sandbox first (fastest, always up-to-date)
    sandbox = agent_service.sandboxes.get(id)
    if sandbox:
        try:
            files = await _list_sandbox_files(sandbox)
            return {
                "project_id": id,
                "files": files,
                "sandbox_id": sandbox.sandbox_id,
                "sandbox_active": True,
            }
        except Exception as e:
            print(f"Live sandbox file listing failed for {id}, falling back to disk: {e}")

    # Sandbox not in memory or query failed — fall back to disk snapshot
    metadata_file = os.path.join(agent_service.storage_base_path, id, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file) as f:
            metadata = json.load(f)
        return {
            "project_id": id,
            "files": metadata.get("files", []),
            "sandbox_id": metadata.get("sandbox_id"),
            "sandbox_active": False,
        }
    raise HTTPException(status_code=404, detail="Project not found or no files available.")


@app.get("/projects/{id}/files/{file_path:path}")
async def get_file_content(
    id: str,
    file_path: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the content of a specific file from the project"""
    result = await db.execute(select(Chat).where(Chat.id == id))
    chat = result.scalar_one_or_none()
    if not chat or chat.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")

    # Try live sandbox first
    sandbox = agent_service.sandboxes.get(id)
    if sandbox:
        try:
            full_path = f"/home/user/react-app/{file_path}"
            content = await sandbox.files.read(full_path)
            return {"file_path": file_path, "content": content}
        except Exception as e:
            print(f"Live sandbox read failed for {file_path}, falling back to disk: {e}")

    # Fall back to disk snapshot (sandbox expired or read failed)
    disk_file = os.path.join(
        agent_service.storage_base_path, id, file_path.replace("/", "_")
    )
    if os.path.exists(disk_file):
        with open(disk_file, "r", encoding="utf-8") as f:
            content = f.read()
        return {"file_path": file_path, "content": content}

    raise HTTPException(status_code=404, detail="File not found in sandbox or disk snapshot.")


@app.post("/projects/{id}/restart")
async def restart_project_sandbox(
    id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Recreate the E2B sandbox for an expired project, restore files, and restart Vite."""
    result = await db.execute(select(Chat).where(Chat.id == id))
    chat = result.scalar_one_or_none()
    if not chat or chat.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")

    try:
        # Recreate (or reconnect to) the sandbox and restore files from disk
        sandbox = await agent_service.get_e2b_sandbox(id)

        # Ensure Vite is running — restoration doesn't start it automatically.
        # pkill is best-effort; nohup forks to background so E2B returns
        # exit code -1 (detached process) — both are safe to ignore.
        try:
            await sandbox.commands.run(
                "pkill -f vite || true", cwd="/home/user/react-app"
            )
        except Exception:
            pass
        try:
            await sandbox.commands.run(
                "nohup npm run dev -- --host 0.0.0.0 > /tmp/vite.log 2>&1 &",
                cwd="/home/user/react-app",
            )
        except Exception:
            pass
        await asyncio.sleep(10)

        new_url = f"https://5173-{sandbox.sandbox_id}.e2b.app"

        # Persist the new URL
        chat.app_url = new_url
        await db.commit()

        return {"app_url": new_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart sandbox: {str(e)}")


@app.get("/projects/{id}/download")
async def download_all_files(
    id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Download all project files as a ZIP archive"""
    result = await db.execute(select(Chat).where(Chat.id == id))
    chat = result.scalar_one_or_none()
    if not chat or chat.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")

    sandbox = agent_service.sandboxes.get(id)
    if not sandbox:
        raise HTTPException(status_code=404, detail="Project sandbox not found or not active.")

    try:
        files = await _list_sandbox_files(sandbox)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in files:
                try:
                    content = await sandbox.files.read(f"/home/user/react-app/{file_path}")
                    zip_file.writestr(file_path, content)
                except Exception as e:
                    print(f"Failed to add {file_path} to ZIP: {e}")

        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={id}-project.zip"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating ZIP: {str(e)}")


@app.delete("/projects/{id}")
async def delete_project(
    id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a project — cancels active run, kills sandbox, removes disk files and DB records."""
    result = await db.execute(select(Chat).where(Chat.id == id))
    chat = result.scalar_one_or_none()
    if not chat or chat.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")

    # 1. Cancel active agent run
    task = active_runs.pop(id, None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    # 2. Close SSE queue (drain any waiting consumers)
    active_streams.pop(id, None)

    # 3. Kill E2B sandbox (best-effort — may already be dead)
    sandbox = agent_service.sandboxes.pop(id, None)
    agent_service.project_timestamps.pop(id, None)
    if sandbox:
        try:
            await sandbox.kill()
        except Exception:
            pass

    # 4. Delete disk snapshot directory
    project_dir = os.path.join(agent_service.storage_base_path, id)
    if os.path.exists(project_dir):
        import shutil
        shutil.rmtree(project_dir, ignore_errors=True)

    # 5. Delete DB records (messages first, then chat)
    await db.execute(delete(Message).where(Message.chat_id == id))
    await db.delete(chat)
    await db.commit()

    return {"status": "deleted", "project_id": id}


@app.post("/chats/{id}/cancel")
async def cancel_build(
    id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Cancel an active build for a project."""
    result = await db.execute(select(Chat).where(Chat.id == id))
    chat = result.scalar_one_or_none()
    if not chat or chat.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    task = active_runs.pop(id, None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    # Send a cancelled event so the frontend knows
    event_queue = active_streams.get(id)
    if event_queue:
        await event_queue.put({"e": "cancelled", "message": "Build cancelled by user"})

    return {"status": "cancelled"}


@app.get("/projects", response_model=ProjectsListResponse)
async def list_user_projects(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all Projects per user"""
    result = await db.execute(
        select(Chat).where(Chat.user_id == current_user.id).order_by(Chat.created_at.desc())
    )
    projects = result.scalars().all()
    return ProjectsListResponse(projects=projects)


@app.get("/sse/{id}")
async def sse_stream(
    id: str,
    token: str = Query(...),
):
    """Server-Sent Events endpoint for streaming agent updates with JWT authentication"""

    # Validate JWT token
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    # Verify user and chat ownership before streaming starts
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.id == int(user_id)))
        user = result.scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        result = await db.execute(select(Chat).where(Chat.id == id))
        chat = result.scalar_one_or_none()
        if chat is None:
            raise HTTPException(status_code=404, detail="Chat not found")
        if chat.user_id != user.id:
            raise HTTPException(status_code=403, detail="Unauthorized: Chat belongs to another user")

        chat_app_url = chat.app_url

    # Reuse existing queue if agent is already running, otherwise create new
    event_queue = active_streams.get(id)
    if not event_queue:
        event_queue = asyncio.Queue()
        active_streams[id] = event_queue

    async def event_generator():
        """Generate SSE events from the queue"""
        try:
            # Fetch message history in a short-lived session
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(Message)
                    .where(Message.chat_id == id)
                    .order_by(Message.created_at)
                )
                messages = result.scalars().all()

                print(f"SSE: Sending history with {len(messages)} messages and app_url: {chat_app_url}")

                history_event = {
                    "e": "history",
                    "messages": [
                        {
                            "id": msg.id,
                            "role": msg.role,
                            "content": msg.content,
                            "event_type": msg.event_type,
                            "created_at": msg.created_at.isoformat(),
                            "tool_calls": msg.tool_calls if hasattr(msg, 'tool_calls') else None
                        }
                        for msg in messages
                    ],
                    "app_url": chat_app_url
                }

            yield f"data: {json.dumps(history_event)}\n\n"

            # Stream live events from queue
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=30.0)
                    print(f"SSE: Sending event for {id}: {event.get('e')}")
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    print(f"SSE: Sending keep-alive for {id}")
                    yield ": keep-alive\n\n"

        except asyncio.CancelledError:
            print(f"SSE connection closed for {id}")
        except Exception as e:
            print(f"Error in SSE event generator for {id}: {e}")
            traceback.print_exc()
        finally:
            active_streams.pop(id, None)
            print(f"SSE stream cleaned up for {id}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.post("/chats/{id}/messages")
async def send_message(
    id: str,
    payload: ChatMessagePayload,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Send a new message to the chat and start agent processing"""

    prompt = payload.prompt

    if not prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")

    # Verify chat exists and belongs to user
    result = await db.execute(select(Chat).where(Chat.id == id))
    chat = result.scalar_one_or_none()
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    if chat.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Unauthorized: Chat belongs to another user")

    # Check if agent is already running
    if id in active_runs:
        raise HTTPException(
            status_code=409,
            detail="Project is being created. Please wait for the current build to complete."
        )

    # Store user message
    user_message = Message(
        id=str(uuid.uuid4()),
        chat_id=id,
        role="user",
        content=prompt
    )
    db.add(user_message)
    await db.commit()

    # Get or create event queue
    event_queue = active_streams.get(id)
    if not event_queue:
        event_queue = asyncio.Queue()
        active_streams[id] = event_queue

    # Decrypt user's OpenRouter API key for follow-up messages
    if not current_user.encrypted_openrouter_key:
        raise HTTPException(status_code=400, detail="Please add your OpenRouter API key in Settings before building.")
    openrouter_api_key = decrypt_api_key(current_user.encrypted_openrouter_key)

    # Read model_choice from the chat record so follow-ups use the same model
    chat_model_choice = getattr(chat, "model_choice", DEFAULT_BUILDER_MODEL) or DEFAULT_BUILDER_MODEL

    async def agent_task():
        try:
            await agent_service.run_agent_stream(
                prompt=prompt, id=id, event_queue=event_queue,
                openrouter_api_key=openrouter_api_key, builder_model=chat_model_choice,
                is_first_message=False,
            )
        except Exception as e:
            print(f"Error in agent task for project {id}: {e}")
            traceback.print_exc()

            try:
                async with AsyncSessionLocal() as error_db:
                    error_message = Message(
                        id=str(uuid.uuid4()),
                        chat_id=id,
                        role="assistant",
                        content=f"Build failed: {str(e)}",
                        event_type="error"
                    )
                    error_db.add(error_message)
                    await error_db.commit()
            except Exception as db_err:
                print(f"Failed to store error message: {db_err}")

            try:
                event_queue.put_nowait({
                    "e": "error",
                    "message": f"Build failed: {str(e)}"
                })
            except Exception as queue_err:
                print(f"Failed to send error to event queue: {queue_err}")
        finally:
            active_runs.pop(id, None)

    active_runs[id] = asyncio.create_task(agent_task())

    return {
        "status": "accepted",
        "message": "Message received and agent started",
    }
