from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
import asyncio
from typing import List
from .graph_state import GraphState
from .tools import create_tools, check_missing_packages_standalone
from .agent import create_llm, get_fast_model
from .formatters import (
    create_formatted_message,
    format_plan_as_markdown,
    generate_description_from_plan,
    generate_build_summary,
)
import json
from langgraph.prebuilt import create_react_agent
from .prompts import (
    ENHANCER_PLANNER_PROMPT,
    BUILDER_SYSTEM_FIRST,
    BUILDER_SYSTEM_FOLLOWUP,
    FIXER_PROMPT,
    get_builder_prompt,
    get_fixer_prompt,
)
from utils.store import load_json_store
import traceback
from db.base import AsyncSessionLocal
from db.models import Message
import uuid


class PlanSchema(BaseModel):
    """Validated structure for the planner's output."""
    overview: str = Field(description="Application overview and purpose")
    components: List[str] = Field(default_factory=list, description="React components to create")
    pages: List[str] = Field(default_factory=list, description="Pages / routes")
    dependencies: List[str] = Field(default_factory=list, description="npm packages to install")
    file_structure: List[str] = Field(default_factory=list, description="File paths to create")
    implementation_steps: List[str] = Field(default_factory=list, description="Ordered build steps")


def safe_send_event(event_queue: asyncio.Queue, data: dict):
    """Helper to safely send events to queue"""
    if event_queue:
        try:
            event_queue.put_nowait(data)
        except Exception as e:
            print(f"Event queue send failed: {e}")


async def store_message(chat_id: str, role: str, content: str, event_type: str = None, tool_calls: list = None):
    """Helper to store a message in the database"""
    try:
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
    except Exception as e:
        print(f"Failed to store message: {e}")


async def _stream_agent_events(
    agent_executor,
    messages: list,
    config: dict,
    event_queue: asyncio.Queue,
    project_id: str,
) -> None:
    """Stream agent events to the SSE queue.

    Tool calls are collected in-memory and flushed as a single DB row at the end.
    Real-time SSE events are still sent for every tool_start / tool_end.
    """

    thinking_sent = False
    tool_log: list[dict] = []

    async for event in agent_executor.astream_events(
        {"messages": messages}, version="v2", config=config
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, str):
                            text_parts.append(block)
                        elif isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif hasattr(block, "text"):
                            text_parts.append(block.text)
                    content = "\n".join(filter(None, text_parts))
                else:
                    content = str(content)

                if content:
                    if not thinking_sent:
                        safe_send_event(event_queue, {"e": "thinking", "message": content})
                        thinking_sent = True

        elif kind == "on_tool_start":
            tool_name = event.get("name")
            tool_input = event.get("data", {}).get("input", {})
            safe_send_event(event_queue, {
                "e": "tool_started",
                "tool_name": tool_name,
                "tool_input": tool_input,
            })

        elif kind == "on_tool_end":
            tool_name = event.get("name")
            tool_output = event.get("data", {}).get("output")

            if hasattr(tool_output, "content"):
                tool_output = tool_output.content
            elif not isinstance(tool_output, str):
                tool_output = str(tool_output)

            safe_send_event(event_queue, {
                "e": "tool_completed",
                "tool_name": tool_name,
                "tool_output": tool_output,
            })

            tool_log.append({"name": tool_name, "status": "success", "output": tool_output[:150]})

    # Flush all tool calls as ONE DB row
    if tool_log:
        await store_message(
            chat_id=project_id,
            role="assistant",
            content=f"Executed {len(tool_log)} tool calls: {', '.join(t['name'] for t in tool_log)}",
            event_type="tool_summary",
            tool_calls=tool_log,
        )


# ─────────────────────────────────────────────────────────────
# Node 1: PLANNER-BUILDER (replaces enhancer + planner + builder)
# ─────────────────────────────────────────────────────────────

async def planner_builder_node(state: GraphState, config: RunnableConfig) -> dict:
    """Combined planner + builder node.

    1. Emits enhancer_started / planner_started SSE events
    2. Generates plan via structured output (fast model)
    3. Emits planner_complete + description
    4. Runs builder ReAct agent (expensive model)
    5. Emits builder_complete + summary
    """
    configurable = config.get("configurable", {})
    event_queue = configurable.get("event_queue")
    sandbox = configurable.get("sandbox")

    try:
        if not sandbox:
            raise Exception("Sandbox not available")

        is_first_message = state.get("is_first_message", True)
        user_prompt = state.get("user_prompt", "")
        project_id = state.get("project_id", "")
        api_key = configurable.get("openrouter_api_key")
        builder_model = state.get("builder_model", "google/gemini-2.5-pro")
        fast_model = get_fast_model(builder_model)

        # ── Phase 1: Planning ────────────────────────────────

        safe_send_event(event_queue, {"e": "enhancer_started", "message": "Understanding your idea..."})
        safe_send_event(event_queue, {"e": "planner_started", "message": "Planning the application architecture..."})

        # Load previous context for follow-ups
        previous_context = ""
        if project_id and not is_first_message:
            context = await asyncio.to_thread(load_json_store, project_id, "context.json")
            if context:
                conv_text = ""
                for i, conv in enumerate(context.get("conversation_history", [])[-5:], 1):
                    status = "[OK]" if conv.get("success") else "[FAIL]"
                    conv_text += f"  {i}. {status} {conv.get('user_prompt', '')[:80]}\n"

                previous_context = (
                    f"\n\nPREVIOUS PROJECT CONTEXT:"
                    f"\n{context.get('semantic', '')}"
                    f"\nFiles: {len(context.get('files_created', []))} created"
                    f"\nHistory:\n{conv_text}"
                )

        planner_llm = create_llm(api_key, fast_model, max_tokens=4000)

        plan_messages = [
            SystemMessage(content=ENHANCER_PLANNER_PROMPT),
            HumanMessage(content=f"{user_prompt}{previous_context}"),
        ]

        try:
            structured_llm = planner_llm.with_structured_output(PlanSchema)
            plan_response = await structured_llm.ainvoke(plan_messages)
            plan = plan_response.dict() if hasattr(plan_response, "dict") else plan_response.model_dump()
        except Exception:
            # Fallback: raw text → parse as JSON
            raw_response = await planner_llm.ainvoke(plan_messages)
            raw_text = raw_response.content.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            try:
                plan = json.loads(raw_text)
            except json.JSONDecodeError:
                plan = {
                    "overview": user_prompt,
                    "components": [],
                    "pages": [],
                    "dependencies": [],
                    "file_structure": [],
                    "implementation_steps": [f"Build the application: {user_prompt}"],
                }

        print(f"Plan generated: {json.dumps(plan, indent=2)[:500]}")

        # Emit plan events for frontend
        formatted_plan = create_formatted_message("planner_complete", plan)
        safe_send_event(event_queue, formatted_plan)

        description = generate_description_from_plan(plan, user_prompt)
        safe_send_event(event_queue, {"e": "description", "message": description})

        await store_message(chat_id=project_id, role="assistant", content=description, event_type="description")

        # ── Phase 2: Building ────────────────────────────────

        safe_send_event(event_queue, {"e": "builder_started", "message": "Generating code for your application..."})

        files_tracker: list = []
        tool_mode = "first_build" if is_first_message else "follow_up"
        tools = create_tools(
            sandbox, event_queue, project_id,
            files_tracker=files_tracker,
            mode=tool_mode,
        )

        system_prompt = BUILDER_SYSTEM_FIRST if is_first_message else BUILDER_SYSTEM_FOLLOWUP
        builder_llm = create_llm(api_key, builder_model, max_tokens=16000)

        agent_executor = create_react_agent(
            builder_llm,
            tools,
            prompt=SystemMessage(content=system_prompt),
        )

        user_message = get_builder_prompt(plan, is_first_message)
        messages = [HumanMessage(content=user_message)]

        try:
            await asyncio.wait_for(
                _stream_agent_events(agent_executor, messages, config, event_queue, project_id),
                timeout=600,
            )
        except asyncio.TimeoutError:
            print("Builder agent timed out after 10 minutes")
        except Exception as e:
            print(f"Builder agent error: {e}")
            traceback.print_exc()

        # Emit build summary
        summary = generate_build_summary(files_tracker, [], plan)
        safe_send_event(event_queue, {"e": "summary", "message": summary})
        safe_send_event(event_queue, {"e": "builder_complete", "message": "Build phase complete"})

        await store_message(chat_id=project_id, role="assistant", content=summary, event_type="summary")

        return {
            "plan": plan,
            "files_created": files_tracker,
            "current_node": "planner_builder",
            "execution_log": [{"node": "planner_builder", "status": "completed", "files_created": files_tracker}],
        }

    except Exception as e:
        error_msg = f"Planner-builder error: {str(e)}"
        print(error_msg)
        traceback.print_exc()

        if event_queue:
            safe_send_event(event_queue, {"e": "builder_error", "message": error_msg})

        return {
            "plan": state.get("plan"),
            "files_created": [],
            "current_node": "planner_builder",
            "error_message": error_msg,
            "execution_log": [{"node": "planner_builder", "status": "error", "error": error_msg}],
        }


# ─────────────────────────────────────────────────────────────
# Node 2: BUILD CHECKPOINT (deterministic — no LLM)
# ─────────────────────────────────────────────────────────────

async def build_checkpoint_node(state: GraphState, config: RunnableConfig) -> dict:
    """Deterministic build check. Runs vite build + auto-installs missing packages.

    No LLM call. Zero token cost.
    """
    configurable = config.get("configurable", {})
    event_queue = configurable.get("event_queue")
    sandbox = configurable.get("sandbox")

    safe_send_event(event_queue, {"e": "code_validator_started", "message": "Validating and fixing any issues..."})

    try:
        if not sandbox:
            raise Exception("Sandbox not available")

        path = "/home/user/react-app"

        # Step 1: Auto-install missing packages
        missing = await check_missing_packages_standalone(sandbox)
        if missing:
            print(f"Build checkpoint: installing missing packages: {missing}")
            safe_send_event(event_queue, {"e": "missing_dependencies", "packages": missing})
            install_cmd = f"npm install {' '.join(missing)}"
            await sandbox.commands.run(install_cmd, cwd=path, timeout=120)

        # Step 2: Clean Vite cache
        await sandbox.commands.run("rm -rf node_modules/.vite-temp", cwd=path, timeout=10)

        # Step 3: Run vite build
        result = await asyncio.wait_for(
            sandbox.commands.run("npx vite build --mode development 2>&1", cwd=path, timeout=120),
            timeout=130,
        )

        build_passed = result.exit_code == 0
        build_errors = "" if build_passed else (result.stderr or result.stdout or "Unknown build error")

        if build_passed:
            print("Build checkpoint: PASSED")
            safe_send_event(event_queue, {"e": "validation_success", "message": "Build passed"})
        else:
            # Truncate errors to last 40 lines for the fixer
            error_lines = build_errors.strip().split("\n")
            build_errors = "\n".join(error_lines[-40:])
            print(f"Build checkpoint: FAILED\n{build_errors[:500]}")
            safe_send_event(event_queue, {"e": "build_test_failed", "message": "Build failed", "error": build_errors[:500]})

        safe_send_event(event_queue, {"e": "code_validator_complete", "message": "Validation complete"})

        return {
            "build_passed": build_passed,
            "build_errors": build_errors,
            "current_node": "build_checkpoint",
            "execution_log": [{"node": "build_checkpoint", "status": "passed" if build_passed else "failed"}],
        }

    except Exception as e:
        error_msg = f"Build checkpoint error: {str(e)}"
        print(error_msg)
        safe_send_event(event_queue, {"e": "code_validator_complete", "message": error_msg})

        return {
            "build_passed": False,
            "build_errors": error_msg,
            "current_node": "build_checkpoint",
            "execution_log": [{"node": "build_checkpoint", "status": "error", "error": error_msg}],
        }


# ─────────────────────────────────────────────────────────────
# Node 3: FIXER (cheap model, only on build failure)
# ─────────────────────────────────────────────────────────────

async def fixer_node(state: GraphState, config: RunnableConfig) -> dict:
    """Fix build errors using a cheap model (Flash/Haiku).

    Only invoked when build_checkpoint fails. Gets 3 tools: read_file, create_file, execute_command.
    Max 2 retries controlled by the graph edge function.
    """
    configurable = config.get("configurable", {})
    event_queue = configurable.get("event_queue")
    sandbox = configurable.get("sandbox")

    fixer_retries = state.get("fixer_retries", 0)

    try:
        if not sandbox:
            raise Exception("Sandbox not available")

        build_errors = state.get("build_errors", "Unknown error")
        project_id = state.get("project_id", "")
        api_key = configurable.get("openrouter_api_key")
        builder_model = state.get("builder_model", "google/gemini-2.5-pro")
        fast_model = get_fast_model(builder_model)

        print(f"Fixer node: attempt {fixer_retries + 1}, fixing errors:\n{build_errors[:300]}")

        tools = create_tools(sandbox, event_queue, project_id, mode="fixer")
        fixer_llm = create_llm(api_key, fast_model, max_tokens=8000)

        agent_executor = create_react_agent(
            fixer_llm,
            tools,
            prompt=SystemMessage(content=FIXER_PROMPT),
        )

        user_message = get_fixer_prompt(build_errors)
        messages = [HumanMessage(content=user_message)]

        try:
            await asyncio.wait_for(
                _stream_agent_events(agent_executor, messages, config, event_queue, project_id),
                timeout=300,
            )
        except asyncio.TimeoutError:
            print("Fixer agent timed out after 5 minutes")
        except Exception as e:
            print(f"Fixer agent error: {e}")
            traceback.print_exc()

        return {
            "fixer_retries": fixer_retries + 1,
            "current_node": "fixer",
            "execution_log": [{"node": "fixer", "status": "completed", "attempt": fixer_retries + 1}],
        }

    except Exception as e:
        error_msg = f"Fixer node error: {str(e)}"
        print(error_msg)

        return {
            "fixer_retries": fixer_retries + 1,
            "current_node": "fixer",
            "error_message": error_msg,
            "execution_log": [{"node": "fixer", "status": "error", "error": error_msg}],
        }


# ─────────────────────────────────────────────────────────────
# Node 4: APP START (deterministic — no LLM)
# ─────────────────────────────────────────────────────────────

async def app_start_node(state: GraphState, config: RunnableConfig) -> dict:
    """Ensure the Vite dev server is running and serving.

    No LLM call. Just port checks and polling.
    """
    configurable = config.get("configurable", {})
    event_queue = configurable.get("event_queue")
    sandbox = configurable.get("sandbox")

    safe_send_event(event_queue, {"e": "app_check_started", "message": "Starting your app and running final checks..."})

    try:
        if not sandbox:
            raise Exception("Sandbox not available")

        path = "/home/user/react-app"
        runtime_errors = []

        # Check essential files
        main_files = ["src/App.jsx", "src/main.jsx", "package.json"]
        missing_files = []
        for file_path in main_files:
            try:
                await sandbox.files.read(f"{path}/{file_path}")
            except Exception:
                missing_files.append(file_path)

        if missing_files:
            runtime_errors.append(f"Missing essential files: {', '.join(missing_files)}")
        else:
            # Check if Vite is listening on port 5173
            port_check = await sandbox.commands.run(
                "ss -tlnp 2>/dev/null | grep -q ':5173' && echo 'port_open' || echo 'port_closed'",
                cwd=path,
            )
            if "port_closed" in port_check.stdout:
                print("App start: Port 5173 not listening — restarting Vite")
                await sandbox.commands.run(
                    "nohup npm run dev -- --host 0.0.0.0 > /tmp/vite.log 2>&1 &",
                    cwd=path,
                )
                # Poll for Vite to start
                for attempt in range(15):
                    await asyncio.sleep(2)
                    poll = await sandbox.commands.run(
                        "ss -tlnp 2>/dev/null | grep -q ':5173' && echo 'ready' || echo 'waiting'",
                        cwd=path,
                    )
                    if "ready" in poll.stdout:
                        print(f"App start: Vite ready after {(attempt + 1) * 2}s")
                        break
                else:
                    print("App start: Vite did not start within 30s")
            else:
                print("App start: Port 5173 already open")

            # Verify HTTP response
            vite_result = await sandbox.commands.run(
                "curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://localhost:5173",
                cwd=path,
            )
            http_code = vite_result.stdout.strip()
            if http_code == "200":
                print("App start: Vite is responding (HTTP 200)")
            else:
                print(f"App start: Vite returned HTTP {http_code}")
                runtime_errors.append(f"Vite dev server returned HTTP {http_code}")

        success = len(runtime_errors) == 0
        safe_send_event(event_queue, {
            "e": "app_check_complete",
            "errors": runtime_errors,
            "message": "App is ready" if success else f"App check found {len(runtime_errors)} issues",
        })

        return {
            "success": success,
            "current_node": "app_start",
            "error_message": "; ".join(runtime_errors) if runtime_errors else None,
            "execution_log": [{"node": "app_start", "status": "completed", "errors": runtime_errors}],
        }

    except Exception as e:
        error_msg = f"App start error: {str(e)}"
        print(error_msg)
        safe_send_event(event_queue, {"e": "app_check_error", "message": error_msg})

        return {
            "success": False,
            "current_node": "app_start",
            "error_message": error_msg,
            "execution_log": [{"node": "app_start", "status": "error", "error": error_msg}],
        }
