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
                await self.sandboxes[id].set_timeout(1800)
                self.project_timestamps[id] = current_time
                print(f"Extended timeout for existing sandbox: {id}")
                return self.sandboxes[id]
            else:
                # Sandbox expired, clean up
                print(f"Sandbox expired for project {id}, recreating...")
                await self.sandboxes[id].kill()
                del self.sandboxes[id]

        # Create new sandbox
        print(f"Initializing new sandbox for project id = {id}")
        if TEMPLATE_ID:
            print(f"Using custom E2B template: {TEMPLATE_ID}")
            self.sandboxes[id] = await AsyncSandbox.create(
                template=TEMPLATE_ID, timeout=1800
            )
        else:
            print("Using default E2B template (no custom template configured)")
            self.sandboxes[id] = await AsyncSandbox.create(
                timeout=1800
            )
        await self.sandboxes[id].set_timeout(1800)
        self.project_timestamps[id] = current_time
        print("Sandbox created with react environment")

        # Restore files if we have them stored on disk
        await self._restore_files_from_disk(id, self.sandboxes[id])

        return self.sandboxes[id]

    async def close_sandbox(self, id: str):
        """Close and cleanup E2B sandbox"""
        if id in self.sandboxes:
            sandbox = self.sandboxes.pop(id)
            await sandbox.kill()
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

        for file_path in files:
            try:
                # Read from disk
                local_file = os.path.join(project_dir, file_path.replace("/", "_"))
                if os.path.exists(local_file):
                    with open(local_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Write to sandbox
                    full_path = f"/home/user/react-app/{file_path}"
                    await sandbox.files.write(full_path, content)
                else:
                    print(f"Local file not found: {local_file}")
            except Exception as e:
                print(f"Failed to restore {file_path}: {e}")

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
        self, project_id: str, user_prompt: str, success: bool
    ):
        """Save conversation history to context for future reference"""
        try:
            

            # Load existing context
            context = load_json_store(project_id, "context.json")

            # Get or initialize conversation history
            conversation_history = context.get("conversation_history", [])

            # Add new conversation entry
            conversation_entry = {
                "timestamp": time.time(),
                "user_prompt": user_prompt,
                "success": success,
                "date": datetime.now().isoformat(),
            }

            conversation_history.append(conversation_entry)

            # Keep only last 10 conversations to avoid bloat
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]

            # Update context
            context["conversation_history"] = conversation_history
            save_json_store(project_id, "context.json", context)

            print(f"Saved conversation history for project {project_id}")

        except Exception as e:
            print(f"Failed to save conversation history: {e}")

    async def snapshot_project_files(self, project_id: str):
        """Snapshot all source files from sandbox to disk"""

        if project_id not in self.sandboxes:
            return

        sandbox = self.sandboxes[project_id]

        project_dir = os.path.join(self.storage_base_path, project_id)
        os.makedirs(project_dir, exist_ok=True)

        paths_to_snapshot = [
            "src",
            "public",
            "package.json",
            "index.html",
        ]

        files_stored = []

        for path in paths_to_snapshot:
            try:
                full_path = f"/home/user/react-app/{path}"
                result = await sandbox.commands.run(
                    f"test -f {full_path} && echo 'file' || test -d {full_path} && echo 'dir'",
                    cwd="/home/user/react-app",
                )

                if "file" in result.stdout:
                    content = await sandbox.files.read(full_path)
                    local_file = os.path.join(project_dir, path.replace("/", "_"))
                    with open(local_file, "w", encoding="utf-8") as f:
                        f.write(content)
                    files_stored.append(path)

                elif "dir" in result.stdout:
                    find_result = await sandbox.commands.run(
                        f"find {path} -type f", cwd="/home/user/react-app"
                    )
                    file_paths = find_result.stdout.strip().split("\n")

                    for file_path in file_paths:
                        if file_path and not file_path.startswith("."):
                            try:
                                content = await sandbox.files.read(
                                    f"/home/user/react-app/{file_path}"
                                )
                                local_file = os.path.join(
                                    project_dir, file_path.replace("/", "_")
                                )
                                with open(local_file, "w", encoding="utf-8") as f:
                                    f.write(content)
                                files_stored.append(file_path)
                            except Exception as e:
                                print(f"Failed to snapshot {file_path}: {e}")
            except Exception as e:
                print(f"Failed to snapshot {path}: {e}")

        # Save metadata
        metadata = {
            "project_id": project_id,
            "files": files_stored,
            "timestamp": time.time(),
        }
        metadata_file = os.path.join(project_dir, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Snapshotted {len(files_stored)} files for project {project_id} to disk")

    async def _store_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        event_type: str = None,
        tool_calls: list = None,
    ):
        """Helper to store a message in the database"""
        async with AsyncSessionLocal() as db:
            message = Message(
                id=str(uuid.uuid4()),
                chat_id=chat_id,
                role=role,
                content=content,
                event_type=event_type,
                tool_calls=tool_calls,
            )
            db.add(message)
            await db.commit()

    def _send_event(self, event_queue: asyncio.Queue, data: dict):
        """Helper to safely send event to queue"""
        try:
            event_queue.put_nowait(data)
        except Exception as e:
            print(f"Failed to queue event: {e}")
            return

    async def run_agent_stream(self, prompt: str, id: str, event_queue: asyncio.Queue):
        """
        Run the LangGraph multi-agent workflow
        """
        try:
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
                "enhanced_prompt": prompt,
                "plan": None,
                "files_created": [],
                "files_modified": [],
                "current_errors": {},
                "validation_errors": [],
                "runtime_errors": [],
                "retry_count": {
                    "validation_errors": 0,
                    "runtime_errors": 0,
                },
                "max_retries": 3,
                "sandbox": sandbox,
                "event_queue": event_queue,
                "current_node": "",
                "execution_log": [],
                "success": False,
                "error_message": None,
            }

            print(f"Starting LangGraph workflow with prompt: {prompt}")
            print(f"Project ID: {id}")

            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)

            # Get the final URL
            host = sandbox.get_host(port=5173)
            url = f"https://{host}"

            print(f"\nWorkflow completed. Project live at: {url}\n")
            print(f"Workflow success: {final_state.get('success')}")
            print(f"Files created: {final_state.get('files_created')}")

            # Save conversation history for future context
            await self._save_conversation_history(
                project_id=id,
                user_prompt=prompt,
                success=final_state.get('success', False)
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
                "files_created": final_state.get('files_created', [])
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