from .graph_builder import get_workflow
from typing import Dict
from e2b_code_interpreter import AsyncSandbox
from dotenv import load_dotenv
import asyncio
from db.base import AsyncSessionLocal
from db.models import Message, Chat
from sqlalchemy import select
import os
import json
import time
import traceback
import uuid
from datetime import datetime
from utils.store import load_json_store, save_json_store

load_dotenv()

# After building your E2B template with 'e2b template build',
# copy the generated template_id from e2b.toml and paste it here
TEMPLATE_ID = os.getenv("E2B_TEMPLATE_ID", None)  # Set in .env or use None for default
base_path = "/home/user/react-app"


class Service:
    """
    LangGraph-based multi-agent service for React application development
    """

    def __init__(self) -> None:
        self.sandboxes: Dict[str, AsyncSandbox] = {}
        self.workflow = get_workflow()
        self.project_timestamps: Dict[str, float] = {}
        self.sandbox_timeout = 1800
        self.storage_base_path = os.path.join(
            os.path.dirname(__file__), "..", "projects"
        )
        os.makedirs(self.storage_base_path, exist_ok=True)

    async def get_e2b_sandbox(self, id: str) -> AsyncSandbox:
        """Get or create E2B sandbox for project"""

        current_time = time.time()

        # Check if sandbox exists and is still valid
        if id in self.sandboxes:
            last_access = self.project_timestamps.get(id, 0)
            time_elapsed = current_time - last_access

            if time_elapsed < self.sandbox_timeout:
                await self.sandboxes[id].set_timeout(self.sandbox_timeout)
                self.project_timestamps[id] = current_time
                print(f"Extended timeout for existing sandbox: {id}")
                return self.sandboxes[id]
            else:
                # Sandbox expired, clean up. The E2B sandbox may already be
                # dead server-side, so kill() is best-effort — don't let it
                # crash the recreation path if it throws.
                print(f"Sandbox expired for project {id}, recreating...")
                try:
                    await self.sandboxes[id].kill()
                except Exception as kill_err:
                    print(f"Failed to kill expired sandbox (likely already dead): {kill_err}")
                del self.sandboxes[id]

        # Cache miss — try to reconnect to an existing E2B sandbox before
        # creating a new one. Reconnect succeeds when the server restarted
        # but the E2B sandbox is still alive within its 30-min TTL.
        sandbox, is_new = await self._try_reconnect_sandbox(id)
        self.sandboxes[id] = sandbox
        await sandbox.set_timeout(self.sandbox_timeout)
        self.project_timestamps[id] = current_time
        print(f"Sandbox ready for project {id} (new={is_new})")

        if is_new:
            # Only restore files into a freshly-created sandbox; a reconnected
            # sandbox already has all files from the previous session.
            await self._restore_files_from_disk(id, sandbox)

        return sandbox

    async def _try_reconnect_sandbox(self, project_id: str) -> tuple:
        """Reconnect to a previous sandbox if its ID is stored in metadata,
        otherwise create a fresh one.

        Returns (sandbox, is_new) where is_new=False means the sandbox is
        already populated with files and does NOT need _restore_files_from_disk.
        """
        metadata_file = os.path.join(self.storage_base_path, project_id, "metadata.json")
        sandbox_id = None
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    stored = json.load(f)
                sandbox_id = stored.get("sandbox_id")
            except Exception as e:
                print(f"Could not read metadata for sandbox reconnect: {e}")

        if sandbox_id:
            try:
                print(f"Attempting to reconnect to sandbox {sandbox_id} for project {project_id}")
                sandbox = await asyncio.wait_for(
                    AsyncSandbox.reconnect(sandbox_id), timeout=30
                )
                print(f"Reconnected to existing sandbox {sandbox_id}")
                return sandbox, False
            except asyncio.TimeoutError:
                print(f"Sandbox reconnect timed out after 30s for {sandbox_id}, creating fresh sandbox")
            except Exception as e:
                print(f"Sandbox reconnect failed (likely expired): {e}")

        # Fallback: create a fresh sandbox
        print(f"Creating new sandbox for project {project_id}")
        if TEMPLATE_ID:
            print(f"Using custom E2B template: {TEMPLATE_ID}")
            sandbox = await AsyncSandbox.create(template=TEMPLATE_ID, timeout=1800)
        else:
            print("Using default E2B template")
            sandbox = await AsyncSandbox.create(timeout=1800)
        return sandbox, True

    async def close_sandbox(self, id: str):
        """Close and cleanup E2B sandbox"""
        if id in self.sandboxes:
            sandbox = self.sandboxes.pop(id)
            try:
                await sandbox.kill()
            except Exception as kill_err:
                print(f"Failed to kill sandbox {id} (likely already dead): {kill_err}")
            print(f"closed sandbox: {id}")

    async def _restore_files_from_disk(self, project_id: str, sandbox: AsyncSandbox):
        """Restore files from disk to sandbox"""

        project_dir = os.path.join(self.storage_base_path, project_id)

        if not os.path.exists(project_dir):
            print(f"No stored files found for project {project_id}")
            return

        metadata_file = os.path.join(project_dir, "metadata.json")
        if not os.path.exists(metadata_file):
            print(f"No metadata found for project {project_id}")
            return

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        files = metadata.get("files", [])
        print(f"Restoring {len(files)} files for project {project_id}")

        async def write_one(file_path: str):
            try:
                local_file = os.path.join(project_dir, file_path.replace("/", "_"))
                if os.path.exists(local_file):
                    with open(local_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    await sandbox.files.write(f"/home/user/react-app/{file_path}", content)
                else:
                    print(f"Local file not found: {local_file}")
            except Exception as e:
                print(f"Failed to restore {file_path}: {e}")

        await asyncio.gather(*[write_one(p) for p in files])
        print(f"File restoration complete for project {project_id}")

        # Clean Vite cache to prevent permission issues
        try:
            await sandbox.commands.run(
                "rm -rf node_modules/.vite-temp", cwd="/home/user/react-app"
            )
            print("Cleaned Vite cache after restoration")
        except Exception as e:
            print(f"Failed to clean Vite cache: {e}")

    async def _save_conversation_history(
        self, project_id: str, user_prompt: str, success: bool,
        files_created: list = None,
    ):
        """Save conversation history and merge newly created files into context.json."""
        try:
            # Load existing context
            context = load_json_store(project_id, "context.json")

            # Update conversation history
            conversation_history = context.get("conversation_history", [])
            conversation_history.append({
                "timestamp": time.time(),
                "user_prompt": user_prompt,
                "success": success,
                "date": datetime.now().isoformat(),
            })
            # Keep only last 10 conversations to avoid bloat
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
            context["conversation_history"] = conversation_history

            # Merge newly created files into the files_created list.
            # Uses dict.fromkeys to deduplicate while preserving insertion order;
            # existing files from previous sessions are kept so the planner always
            # sees the full project file count, not just the current session's files.
            if files_created:
                existing = context.get("files_created", [])
                context["files_created"] = list(dict.fromkeys(existing + files_created))

            save_json_store(project_id, "context.json", context)

            print(f"Saved conversation history for project {project_id}")

        except Exception as e:
            print(f"Failed to save conversation history: {e}")

    async def snapshot_project_files(self, project_id: str):
        """Snapshot all source files from sandbox to disk.

        Uses a single find command + concurrent asyncio.gather reads instead of
        the previous per-path type-check loop, reducing wall time from O(N*rtt)
        to O(1*rtt + N*parallel_rtt).  Also persists sandbox_id so the service
        can reconnect to this sandbox after a server restart instead of creating a new one.
        """
        if project_id not in self.sandboxes:
            return

        sandbox = self.sandboxes[project_id]
        project_dir = os.path.join(self.storage_base_path, project_id)
        os.makedirs(project_dir, exist_ok=True)

        # Collect all relevant file paths with a single shell command
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
            print(f"No files found to snapshot for project {project_id}")
            return

        # Read all files concurrently instead of serially
        async def read_and_save(file_path: str):
            try:
                content = await sandbox.files.read(f"/home/user/react-app/{file_path}")
                local_file = os.path.join(project_dir, file_path.replace("/", "_"))
                with open(local_file, "w", encoding="utf-8") as f:
                    f.write(content)
                return file_path
            except Exception as e:
                print(f"Failed to snapshot {file_path}: {e}")
                return None

        results = await asyncio.gather(*[read_and_save(p) for p in file_paths])
        files_stored = [p for p in results if p is not None]

        # Save metadata — sandbox_id lets _try_reconnect_sandbox reconnect to
        # this sandbox after a server restart instead of creating a new one.
        metadata = {
            "project_id": project_id,
            "sandbox_id": sandbox.sandbox_id,
            "files": files_stored,
            "timestamp": time.time(),
        }
        metadata_file = os.path.join(project_dir, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Snapshotted {len(files_stored)} files for project {project_id} to disk")

    async def _classify_prompt(self, prompt: str, api_key: str, builder_model: str) -> str:
        """Classify whether the prompt is a build request or general chat."""
        from langchain_core.messages import SystemMessage, HumanMessage
        from .prompts import GUARDRAIL_PROMPT
        from .agent import create_llm, get_fast_model

        try:
            fast_model = get_fast_model(builder_model)
            llm = create_llm(api_key, fast_model, max_tokens=16)
            messages = [
                SystemMessage(content=GUARDRAIL_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = await llm.ainvoke(messages)
            classification = response.content.strip().lower()
            print(f"Prompt classification: '{classification}' for: {prompt[:80]}")
            return "build" if "build" in classification else "chat"
        except Exception as e:
            # On classification failure, default to "build" so we never block real requests
            print(f"Classification failed, defaulting to build: {e}")
            return "build"

    async def _handle_chat_response(self, prompt: str, project_id: str, event_queue: asyncio.Queue, api_key: str, builder_model: str):
        """Answer a non-build prompt conversationally instead of running the full pipeline."""
        from langchain_core.messages import SystemMessage, HumanMessage
        from .prompts import CHAT_RESPONSE_PROMPT
        from .agent import create_llm, get_fast_model

        fast_model = get_fast_model(builder_model)
        llm = create_llm(api_key, fast_model, max_tokens=512)
        messages = [
            SystemMessage(content=CHAT_RESPONSE_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = await llm.ainvoke(messages)
        answer = response.content.strip()

        # Send the conversational response via SSE
        event_queue.put_nowait({"e": "chat_response", "message": answer})

        # Store in DB
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

        # Signal completion (no URL since nothing was built)
        event_queue.put_nowait({
            "e": "completed",
            "url": None,
            "success": True,
            "files_created": [],
        })

    async def run_agent_stream(self, prompt: str, id: str, event_queue: asyncio.Queue, openrouter_api_key: str = "", builder_model: str = "google/gemini-2.5-pro", is_first_message: bool = True):
        """
        Run the LangGraph multi-agent workflow
        """
        try:
            # Guardrail: classify prompt before creating sandbox or running workflow
            classification = await self._classify_prompt(prompt, openrouter_api_key, builder_model)
            if classification == "chat":
                print(f"Prompt classified as chat, responding conversationally: {prompt[:80]}")
                await self._handle_chat_response(prompt, id, event_queue, openrouter_api_key, builder_model)
                return

            event_queue.put_nowait(
                {
                    "e": "started",
                    "message": "Starting LangGraph multi-agent workflow",
                }
            )

            sandbox = await self.get_e2b_sandbox(id=id)

            initial_state = {
                "project_id": id,
                "user_prompt": prompt,
                "is_first_message": is_first_message,
                "plan": None,
                "files_created": [],
                "build_passed": False,
                "build_errors": "",
                "fixer_retries": 0,
                "max_fixer_retries": 2,
                "current_node": "",
                "execution_log": [],
                "builder_model": builder_model,
                "success": False,
                "error_message": None,
            }

            print(f"Starting LangGraph workflow with prompt: {prompt}")
            print(f"Project ID: {id}")

            # Run the workflow — sandbox and event_queue travel via config, not state,
            # so they never block LangGraph checkpointing.
            final_state = await self.workflow.ainvoke(
                initial_state,
                config={
                    "configurable": {"sandbox": sandbox, "event_queue": event_queue, "openrouter_api_key": openrouter_api_key},
                    "recursion_limit": 100,
                },
            )

            # Get the final URL
            host = sandbox.get_host(port=5173)
            url = f"https://{host}"

            # files_created accumulates across retries via operator.add;
            # deduplicate while preserving first-seen order before sending.
            all_files = final_state.get('files_created', [])
            unique_files = list(dict.fromkeys(all_files))

            print(f"\nWorkflow completed. Project live at: {url}\n")
            print(f"Workflow success: {final_state.get('success')}")
            print(f"Files created: {unique_files}")

            # Save conversation history for future context, including the
            # deduplicated file list so planner sees correct file count next session.
            await self._save_conversation_history(
                project_id=id,
                user_prompt=prompt,
                success=final_state.get('success', False),
                files_created=unique_files,
            )

            # Snapshot files to local disk for persistence
            await self.snapshot_project_files(project_id=id)

            async with AsyncSessionLocal() as db:
                completion_message = Message(
                    id=str(uuid.uuid4()),
                    chat_id=id,
                    role="assistant",
                    content="LangGraph workflow completed" if final_state.get('success') else f"Workflow completed with errors: {final_state.get('error_message')}",
                    event_type="completed",
                )
                db.add(completion_message)

                result = await db.execute(select(Chat).where(Chat.id == id))
                chat = result.scalar_one_or_none()
                if chat:
                    chat.app_url = url
                    print(f"Saved app_url to database: {url}")

                await db.commit()

            event_queue.put_nowait({
                "e": "completed",
                "url": url,
                "success": final_state.get('success'),
                "files_created": unique_files,
            })

        except Exception as e:
            print(f"Error during LangGraph workflow execution: {e}")
            print(f"Error type: {type(e)}")
            print(f"Error details: {str(e)}")


            traceback.print_exc()

            try:
                event_queue.put_nowait({
                    "e": "error",
                    "message": f"Workflow failed: {str(e)}"
                })
            except Exception as queue_err:
                print(f"Failed to send error to event queue: {queue_err}")


agent_service = Service()
