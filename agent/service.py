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
from .base_template import LOCKED_FILES
from .build_agent import run_build_stream
from .edit_agent import run_edit_stream, run_error_fix_stream
from .sandbox import (
    create_sandbox,
    update_sandbox_files,
    validate_sandbox_build,
)
from .prompts import GUARDRAIL_PROMPT, CHAT_RESPONSE_PROMPT, ENHANCE_PROMPT

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

        # Always force-write locked base template files (especially vite.config.js
        # with allowedHosts: true) to ensure sandbox previews work after restart
        from .base_template import BASE_TEMPLATE
        for locked_file in LOCKED_FILES:
            if locked_file in BASE_TEMPLATE:
                await write_one(locked_file, BASE_TEMPLATE[locked_file])

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

    async def _enhance_prompt(self, prompt: str, api_key: str) -> str:
        """Enhance a vague user prompt into a detailed app specification.
        Skips enhancement if prompt is already detailed (200+ chars).
        """
        if len(prompt.strip()) >= 200:
            print(f"Prompt already detailed ({len(prompt)} chars), skipping enhancement")
            return prompt

        from langchain_core.messages import SystemMessage, HumanMessage
        from .agent import create_edit_llm

        try:
            llm = create_edit_llm(api_key)
            messages = [
                SystemMessage(content=ENHANCE_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = await llm.ainvoke(messages)
            enhanced = response.content.strip()
            if enhanced and len(enhanced) > len(prompt):
                print(f"Prompt enhanced: '{prompt[:60]}' → '{enhanced[:80]}...'")
                return enhanced
            return prompt
        except Exception as e:
            print(f"Prompt enhancement failed, using original: {e}")
            return prompt

    async def _handle_chat_response(self, prompt, project_id, event_queue, api_key):
        """Answer a non-build prompt conversationally."""
        from langchain_core.messages import SystemMessage, HumanMessage
        from .agent import create_edit_llm
        from utils.store import load_all_project_files

        # Load project files for context (if any exist)
        project_files = load_all_project_files(project_id)
        if project_files:
            file_list = "\n".join(f"  - {path}" for path in sorted(project_files.keys()))
            project_context = f"The user has a project with these files:\n{file_list}"
        else:
            project_context = "The user has not built a project yet."

        system_prompt = CHAT_RESPONSE_PROMPT.format(project_context=project_context)

        llm = create_edit_llm(api_key)
        messages = [
            SystemMessage(content=system_prompt),
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
            await sandbox_task

            event_queue.put_nowait({"e": "started"})

            workflow_start = time.time()

            # Enhance prompt if it's short/vague
            if len(prompt.strip()) < 200:
                event_queue.put_nowait({"e": "log", "message": "Enhancing your prompt..."})
            enhanced_prompt = await self._enhance_prompt(prompt, api_key)

            event_queue.put_nowait({"e": "log", "message": "Building your application..."})

            # Run build agent with enhanced prompt
            def on_build_event(event):
                event_queue.put_nowait(event)

            result = await asyncio.wait_for(
                run_build_stream(enhanced_prompt, api_key, on_build_event),
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
            build_files = sorted(f["path"] for f in generated_files)
            event_queue.put_nowait({
                "e": "completed",
                "success": True,
                "url": url,
                "duration_s": duration,
                "files": build_files,
            })

            build_summary = [{
                "name": "build_summary",
                "status": "success",
                "detail": f"{len(generated_files)} files generated",
                "output": json.dumps({
                    "duration_s": duration,
                    "files": build_files,
                }),
            }]
            await self._store_message(project_id, "assistant", f"Build complete. Preview: {url}", "completed", tool_calls=build_summary)

            # Background: snapshot + save history
            asyncio.create_task(self._post_build_cleanup(
                project_id, prompt, True,
                [f["path"] for f in generated_files],
                sandbox_id,
            ))

        except asyncio.CancelledError:
            print(f"Build cancelled for {project_id}")
            event_queue.put_nowait({"e": "cancelled", "message": "Build cancelled by user"})
            raise
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
                try:
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
                except Exception:
                    current_files = {}

            if not current_files:
                # No project exists yet — treat as first build
                print(f"No project files found for {project_id}, falling back to first build")
                await self.handle_first_build(message, api_key, project_id, event_queue)
                return

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

            # Apply changes to current files (skip locked base template files)
            for change in file_changes:
                path = change["path"]
                if path in LOCKED_FILES:
                    print(f"Skipping locked file from edit: {path}")
                    continue
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

                    # Apply fixes (skip locked base template files)
                    for fix in fixes:
                        path = fix["path"]
                        if path in LOCKED_FILES:
                            print(f"Skipping locked file from error fix: {path}")
                            continue
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
            edit_files = sorted(c["path"] for c in file_changes)
            event_queue.put_nowait({
                "e": "completed",
                "success": True,
                "url": url,
                "duration_s": duration,
                "files": edit_files,
            })

            edit_summary = [{
                "name": "build_summary",
                "status": "success",
                "detail": f"{len(file_changes)} files modified",
                "output": json.dumps({
                    "duration_s": duration,
                    "files": sorted(c["path"] for c in file_changes),
                }),
            }]
            await self._store_message(project_id, "assistant", "Edit complete.", "completed", tool_calls=edit_summary)

            # Background cleanup
            asyncio.create_task(self._post_build_cleanup(
                project_id, message, True,
                [c["path"] for c in file_changes],
                None,
            ))

        except asyncio.CancelledError:
            print(f"Edit cancelled for {project_id}")
            event_queue.put_nowait({"e": "cancelled", "message": "Build cancelled by user"})
            raise
        except asyncio.TimeoutError:
            event_queue.put_nowait({"e": "error", "message": "Edit timed out. Please try a simpler request."})
        except Exception as e:
            error_msg = str(e)
            print(f"Edit error: {error_msg}")
            traceback.print_exc()
            event_queue.put_nowait({"e": "error", "message": error_msg})

    # ── Helpers ──

    async def _store_message(self, chat_id, role, content, event_type, tool_calls=None):
        try:
            async with AsyncSessionLocal() as db:
                msg = Message(
                    id=str(uuid.uuid4()),
                    chat_id=chat_id,
                    role=role,
                    content=content,
                    event_type=event_type,
                    tool_calls=tool_calls,
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
