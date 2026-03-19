from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
import asyncio
from typing import List
from .graph_state import GraphState
from .tools import create_tools_with_context
from .agent import create_llm, get_fast_model
from .formatters import (
    create_formatted_message,
    format_plan_as_markdown,
    generate_description_from_plan,
    generate_build_summary,
)
import json
from langgraph.prebuilt import create_react_agent
from .prompts import INITPROMPT, VALIDATOR_PROMPT, ENHANCER_PROMPT, get_builder_error_prompt, get_builder_prompt
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
    """Stream agent events to the SSE queue and store them in the DB."""

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
                    safe_send_event(event_queue, {"e": "thinking", "message": content})
                    if len(content) > 50:
                        await store_message(
                            chat_id=project_id,
                            role="assistant",
                            content=content,
                            event_type="thinking",
                        )

        elif kind == "on_tool_start":
            tool_name = event.get("name")
            tool_input = event.get("data", {}).get("input", {})
            safe_send_event(event_queue, {
                "e": "tool_started",
                "tool_name": tool_name,
                "tool_input": tool_input,
            })
            await store_message(
                chat_id=project_id,
                role="assistant",
                content=f"Using tool: {tool_name}",
                event_type="tool_started",
                tool_calls=[{"name": tool_name, "status": "running", "input": str(tool_input)}],
            )

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
            await store_message(
                chat_id=project_id,
                role="assistant",
                content=f"Completed: {tool_name}\n{tool_output[:200]}",
                event_type="tool_completed",
                tool_calls=[{"name": tool_name, "status": "success", "output": tool_output[:500]}],
            )



async def enhancer_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Enhancer node: Expands short/vague user prompts into detailed product descriptions.
    Only runs on the first message — follow-up prompts pass through unchanged.
    """
    configurable = config.get("configurable", {})
    event_queue = configurable.get("event_queue")

    user_prompt = state.get("user_prompt", "")
    is_first_message = state.get("is_first_message", True)

    # Follow-up prompts: skip enhancement, pass through as-is
    if not is_first_message:
        print("Enhancer node: follow-up prompt, skipping enhancement")
        return {"enhanced_prompt": user_prompt}

    try:
        if event_queue:
            safe_send_event(event_queue, {
                "e": "enhancer_started",
                "message": "Understanding your idea...",
            })

        print(f"Enhancer node: enhancing prompt: {user_prompt[:80]}")

        messages = [
            SystemMessage(content=ENHANCER_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        api_key = configurable.get("openrouter_api_key")
        builder_model = state.get("builder_model", "google/gemini-2.5-pro")
        fast_model = get_fast_model(builder_model)
        llm = create_llm(api_key, fast_model)
        print(f"Enhancer node: Using {fast_model} via OpenRouter")
        response = await llm.ainvoke(messages)
        enhanced = response.content.strip()

        print(f"Enhancer node: enhanced prompt: {enhanced[:120]}")

        if event_queue:
            safe_send_event(event_queue, {
                "e": "enhancer_complete",
                "message": "Got it! Planning your application...",
            })

        return {"enhanced_prompt": enhanced}

    except Exception as e:
        # On any failure, fall back to the original prompt — never block the build
        print(f"Enhancer node failed, using original prompt: {e}")
        return {"enhanced_prompt": user_prompt}


async def planner_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Planner node: Analyzes user prompt and generates comprehensive implementation plan
    """
    configurable = config.get("configurable", {})
    event_queue = configurable.get("event_queue")

    try:
        if event_queue:
            safe_send_event(event_queue,
                {
                    "e": "planner_started",
                    "message": "Planning the application architecture...",
                }
            )

        enhanced_prompt = state.get("enhanced_prompt", state.get("user_prompt", ""))
        project_id = state.get("project_id", "")
        print(f"INFO: Received prompt for project {project_id or '(unknown)'}")
        print(f"INFO: Project ID: {project_id}")

        previous_context = ""

        # check if previous context is there
        if project_id:
            print(f"Loading context for project: {project_id}")
            context = await asyncio.to_thread(load_json_store, project_id, "context.json")
            print(f"Loaded context: {context}")

            if context:
                print(f"Context keys found: {context.keys()}")

                # Format conversation history
                conversation_history_text = ""
                conversation_history = context.get("conversation_history", [])
                print(f"Conversation history entries: {len(conversation_history)}")

                if conversation_history:
                    conversation_history_text = (
                        "\nCONVERSATION HISTORY (Last requests):\n"
                    )
                    for i, conv in enumerate(conversation_history[-5:], 1):
                        status = "[SUCCESS]" if conv.get("success") else "[FAILED]"
                        conversation_history_text += f"   {i}. {status} {conv.get('user_prompt', 'Unknown')[:100]}\n"

                previous_context = f"""

                IMPORTANT: PREVIOUS WORK ON THIS PROJECT

                WHAT THIS PROJECT IS:
                {context.get('semantic', 'Not documented')}

                HOW IT WORKS:
                {context.get('procedural', 'Not documented')}

                WHAT HAS BEEN DONE:
                {context.get('episodic', 'Not documented')}

                EXISTING FILES: {len(context.get('files_created', []))} files already exist
                {conversation_history_text}

                CRITICAL: This is an EXISTING project. Your plan should:
                - Build upon what already exists
                - Consider the conversation history to understand the user's intent
                - Only add/modify what's needed for the new request
                - NOT recreate existing components/pages
                - Integrate with the existing structure
                """
                print(f"Previous context prepared successfully")
            else:
                print("No previous context found - empty dict returned")

        planning_prompt = f"""
        You are an expert React application architect. Analyze the following user request and create a comprehensive implementation plan.
        {previous_context}

        USER REQUEST:
        {enhanced_prompt}

        Create a detailed plan that includes:
        1. Application overview and purpose
        2. Component hierarchy and structure
        3. Page/routing structure
        4. Required dependencies
        5. File structure
        6. Implementation steps

        {"NOTE: Since this is an existing project, focus your plan on the NEW features/changes requested, not recreating everything." if previous_context else ""}

        Respond with a JSON object containing the plan.
        """

        messages = [
            SystemMessage(
                content="You are an expert React application architect. Create detailed implementation plans."
            ),
            HumanMessage(content=planning_prompt),
        ]

        if event_queue:
            safe_send_event(event_queue, {"e": "thinking", "message": "Analyzing your request and creating implementation plan..."})

        # Store thinking message
        await store_message(
            chat_id=state.get("project_id"),
            role="assistant",
            content="Analyzing your request and creating implementation plan...",
            event_type="thinking"
        )

        # Structured output — validates the plan against PlanSchema; no manual JSON parsing needed
        api_key = configurable.get("openrouter_api_key")
        builder_model = state.get("builder_model", "google/gemini-2.5-pro")
        fast_model = get_fast_model(builder_model)
        llm = create_llm(api_key, fast_model)
        print(f"Planner node: Using {fast_model} via OpenRouter")
        plan_result = await llm.with_structured_output(PlanSchema).ainvoke(messages)
        plan = plan_result.model_dump()

        # Create formatted plan message
        formatted_plan_msg = create_formatted_message(
            "planner_complete",
            plan,
            message="Planning completed successfully"
        )

        if event_queue:
            safe_send_event(event_queue, formatted_plan_msg)

        # Store plan completion with formatted markdown
        await store_message(
            chat_id=state.get("project_id"),
            role="assistant",
            content=formatted_plan_msg.get("formatted", json.dumps(plan, indent=2)),
            event_type="planner_complete"
        )

        # Generate and send human-readable description
        description = generate_description_from_plan(plan, enhanced_prompt)
        if description:
            if event_queue:
                safe_send_event(event_queue, {"e": "description", "message": description})
            await store_message(
                chat_id=state.get("project_id"),
                role="assistant",
                content=description,
                event_type="description"
            )

        return {
            "plan": plan,
            "current_node": "planner",
            "execution_log": [{"node": "planner", "status": "completed", "plan": plan}],
        }

    except Exception as e:
        error_msg = f"Planner node error: {str(e)}"
        print(error_msg)

        if event_queue:
            safe_send_event(event_queue, {"e": "planner_error", "message": error_msg})

        return {
            "plan": {},
            "current_node": "planner",
            "error_message": error_msg,
            "execution_log": [{"node": "planner", "status": "error", "error": error_msg}],
        }


async def builder_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Builder node: Creates and modifies files based on plan or feedback
    """
    configurable = config.get("configurable", {})
    event_queue = configurable.get("event_queue")
    sandbox = configurable.get("sandbox")

    try:
        if not sandbox:
            raise Exception("Sandbox not available")

        if event_queue:
            safe_send_event(event_queue,
                {
                    "e": "builder_started",
                    "message": "Starting to build the application...",
                }
            )

        # On first message: patch vite.config.js AND reset Home.jsx to a blank
        # placeholder so the LLM doesn't mistake the starter template for existing
        # project work and exit without writing any files.
        is_first_message = state.get("is_first_message", True)
        if is_first_message:
            vite_config = (
                "import { defineConfig } from 'vite'\n"
                "import react from '@vitejs/plugin-react'\n"
                "import tailwindcss from '@tailwindcss/vite'\n\n"
                "export default defineConfig({\n"
                "  plugins: [react(), tailwindcss()],\n"
                "  server: {\n"
                "    host: true,\n"
                "    allowedHosts: true\n"
                "  }\n"
                "})\n"
            )
            home_placeholder = (
                "// PLACEHOLDER — replace this entire file with the real application\n"
                "export default function Home() {\n"
                "  return <div>Building your app...</div>\n"
                "}\n"
            )
            try:
                await sandbox.files.write("/home/user/react-app/vite.config.js", vite_config)
                print("INFO: Patched vite.config.js to support JSX in .js files")
            except Exception as e:
                print(f"WARNING: Failed to patch vite.config.js: {e}")
            try:
                await sandbox.files.write("/home/user/react-app/src/pages/Home.jsx", home_placeholder)
                print("INFO: Reset Home.jsx to placeholder for clean build")
            except Exception as e:
                print(f"WARNING: Failed to reset Home.jsx: {e}")

        plan = state.get("plan", {})
        if plan:
            print("INFO: Plan received")

        current_errors = state.get("current_errors", {})

        project_id = state.get("project_id", "")

        files_tracker: list = []
        base_tools = create_tools_with_context(sandbox, event_queue, project_id, files_tracker=files_tracker)

        if current_errors:
            error_details = []
            for error_type, errors in current_errors.items():
                if isinstance(errors, list):
                    for err in errors:
                        if isinstance(err, dict):
                            error_msg = err.get("error", str(err))
                            error_details.append(f"ERROR: {error_msg}")
                        else:
                            error_details.append(f"ERROR: {str(err)}")
                else:
                    error_details.append(f"{error_type}: {str(errors)}")

            builder_prompt = get_builder_error_prompt("\n".join(error_details))
        else:
            builder_prompt = get_builder_prompt(plan)

        messages = [
            SystemMessage(content=INITPROMPT),
            HumanMessage(content=builder_prompt),
        ]

        api_key = configurable.get("openrouter_api_key")
        builder_model = state.get("builder_model", "google/gemini-2.5-pro")
        builder_llm = create_llm(api_key, builder_model)
        print(f"Builder node: Using {builder_model} via OpenRouter")
        agent_executor = create_react_agent(builder_llm, tools=base_tools)
        agent_config = {"recursion_limit": 50}

        try:
            print(
                f"Builder node: Starting agent execution with {len(base_tools)} tools"
            )

            await asyncio.wait_for(
                _stream_agent_events(
                    agent_executor, messages, agent_config, event_queue, project_id
                ),
                timeout=600,
            )
            files_created = files_tracker
            files_modified = []

            print(f"Builder node: Agent execution completed")
            print(f"Builder node: Final files_created: {files_created}")

            print(f"INFO : {files_created}")

            if event_queue:
                safe_send_event(event_queue,
                    {
                        "e": "builder_complete",
                        "files_created": files_created,
                        "files_modified": files_modified,
                        "message": "Building completed",
                    }
                )

            # Generate and send human-readable build summary
            summary = generate_build_summary(files_created, files_modified, plan)
            if summary:
                if event_queue:
                    safe_send_event(event_queue, {"e": "summary", "message": summary})
                await store_message(
                    chat_id=state.get("project_id"),
                    role="assistant",
                    content=summary,
                    event_type="summary"
                )

            return {
                "files_created": files_created,
                "files_modified": files_modified,
                "current_errors": {},
                "current_node": "builder",
                "execution_log": [
                    {
                        "node": "builder",
                        "status": "completed",
                        "files_created": files_created,
                        "files_modified": files_modified,
                    }
                ],
            }

        except asyncio.TimeoutError:
            print("Builder agent timed out after 10 minutes")

            if event_queue:
                safe_send_event(event_queue,
                    {
                        "e": "builder_error",
                        "message": "Builder agent timed out after 10 minutes",
                    }
                )

            return {
                "files_created": [],
                "files_modified": [],
                "current_node": "builder",
                "execution_log": [
                    {"node": "builder", "status": "timeout", "files_created": [], "files_modified": []}
                ],
            }

        except Exception as e:
            print(f"Builder agent execution error: {e}")
            traceback.print_exc()

            if event_queue:
                safe_send_event(event_queue,
                    {
                        "e": "builder_error",
                        "message": f"Builder agent execution error: {str(e)}",
                    }
                )

            return {
                "files_created": [],
                "files_modified": [],
                "current_node": "builder",
                "execution_log": [
                    {"node": "builder", "status": "error", "files_created": [], "files_modified": []}
                ],
            }

    except Exception as e:
        error_msg = f"Builder node error: {str(e)}"
        print(error_msg)

        if event_queue:
            safe_send_event(event_queue, {"e": "builder_error", "message": error_msg})

        return {
            "current_node": "builder",
            "error_message": error_msg,
            "execution_log": [{"node": "builder", "status": "error", "error": error_msg}],
        }



async def code_validator_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Code Validator node: Active React agent that reviews, validates, and fixes code
    """
    configurable = config.get("configurable", {})
    event_queue = configurable.get("event_queue")
    sandbox = configurable.get("sandbox")

    try:
        if not sandbox:
            raise Exception("Sandbox not available")

        if event_queue:
            safe_send_event(event_queue,
                {
                    "e": "code_validator_started",
                    "message": "Code validator agent reviewing and fixing code...",
                }
            )

        project_id = state.get("project_id", "")
        validation_results = {"errors": [], "summary": ""}
        base_tools = create_tools_with_context(
            sandbox, event_queue, project_id,
            validation_results=validation_results,
            include_test_build=False,
        )

        messages = [
            SystemMessage(content=VALIDATOR_PROMPT),
            HumanMessage(content="Begin your validation pass now. Start with check_missing_packages()."),
        ]

        api_key = configurable.get("openrouter_api_key")
        builder_model = state.get("builder_model", "google/gemini-2.5-pro")
        fast_model = get_fast_model(builder_model)
        validator_llm = create_llm(api_key, fast_model)
        print(f"Validator node: Using {fast_model} via OpenRouter")
        validator_agent = create_react_agent(validator_llm, tools=base_tools)
        agent_config = {"recursion_limit": 50}

        try:
            print(
                f"Code validator: Starting agent execution with {len(base_tools)} tools"
            )

            await asyncio.wait_for(
                _stream_agent_events(
                    validator_agent, messages, agent_config, event_queue, project_id
                ),
                timeout=600,
            )

            # Read what the agent actually reported via report_validation_result
            raw_errors = validation_results.get("errors", [])
            validation_errors = [
                {"type": "code_error", "error": str(e)} for e in raw_errors
            ]

            print(f"Code validator: Agent execution completed")
            print(f"Code validator: {len(validation_errors)} errors remain after fixes")

            if event_queue:
                safe_send_event(event_queue,
                    {
                        "e": "validation_success",
                        "message": "Code validator completed - code review and dependencies checked!",
                    }
                )

            update: dict = {
                "validation_errors": validation_errors,
                "current_node": "code_validator",
                "execution_log": [
                    {
                        "node": "code_validator",
                        "status": "completed",
                        "validation_errors": validation_errors,
                    }
                ],
            }

            if validation_errors:
                current_retry = state.get("retry_count", {}).get("validation_errors", 0)
                update["retry_count"] = {
                    **state.get("retry_count", {}),
                    "validation_errors": current_retry + 1,
                }
                update["current_errors"] = {"validation_errors": validation_errors}
                print(f"Code validator: Found {len(validation_errors)} validation errors")
            else:
                print("Code validator: No validation errors found")

            if event_queue:
                safe_send_event(event_queue,
                    {
                        "e": "code_validator_complete",
                        "errors": validation_errors,
                        "message": f"Code validation completed. Found {len(validation_errors)} errors.",
                    }
                )

            return update

        except asyncio.TimeoutError:
            print("Code validator agent timed out after 10 minutes")

            if event_queue:
                safe_send_event(event_queue,
                    {
                        "e": "code_validator_timeout",
                        "message": "Code validator timed out",
                    }
                )

            timeout_errors = [
                {"type": "timeout", "error": "Code validator timed out", "details": "Validation took too long"}
            ]
            current_retry = state.get("retry_count", {}).get("validation_errors", 0)
            return {
                "validation_errors": timeout_errors,
                "current_node": "code_validator",
                "retry_count": {
                    **state.get("retry_count", {}),
                    "validation_errors": current_retry + 1,
                },
                "current_errors": {"validation_errors": timeout_errors},
                "execution_log": [
                    {"node": "code_validator", "status": "timeout", "validation_errors": timeout_errors}
                ],
            }

    except Exception as e:
        error_msg = f"Code validator node error: {str(e)}"
        print(error_msg)
        traceback.print_exc()

        if event_queue:
            safe_send_event(event_queue, {"e": "code_validator_error", "message": error_msg})

        crash_errors = [{"type": "validator_error", "error": str(e), "details": "Code validator crashed"}]
        current_retry = state.get("retry_count", {}).get("validation_errors", 0)
        return {
            "current_node": "code_validator",
            "error_message": error_msg,
            "validation_errors": crash_errors,
            "retry_count": {
                **state.get("retry_count", {}),
                "validation_errors": current_retry + 1,
            },
            "current_errors": {"validation_errors": crash_errors},
            "execution_log": [{"node": "code_validator", "status": "error", "error": error_msg}],
        }


async def application_checker_node(state: GraphState, config: RunnableConfig) -> dict:
    """
    Application Checker node: Checks if the application is running and captures errors
    """
    configurable = config.get("configurable", {})
    event_queue = configurable.get("event_queue")
    sandbox = configurable.get("sandbox")

    try:
        if not sandbox:
            raise Exception("Sandbox not available")

        if event_queue:
            safe_send_event(event_queue,
                {
                    "e": "app_check_started",
                    "message": "Checking application status and capturing errors...",
                }
            )

        runtime_errors = []

        try:
            # Stage 1: essential files exist
            main_files = ["src/App.jsx", "src/main.jsx", "package.json"]
            missing_files = []

            for file_path in main_files:
                try:
                    await sandbox.files.read(f"/home/user/react-app/{file_path}")
                except Exception:
                    missing_files.append(file_path)

            if missing_files:
                runtime_errors.append(
                    {
                        "type": "missing_files",
                        "error": f"Missing essential files: {', '.join(missing_files)}",
                    }
                )
            else:
                print("Application checker: All essential files present")

                # Stage 2: Vite dev server is actually serving
                # The sandbox runs Vite on port 5173 continuously; if it crashed
                # (e.g. due to a JSX syntax error that slipped past the validator)
                # curl will get a non-200 or a connection-refused.
                try:
                    # Ensure Vite is actually serving on port 5173.
                    # npm install during the build kills the Vite process;
                    # pgrep-based checks give false positives (matches npm cache entries),
                    # so check the port directly instead.
                    port_check = await sandbox.commands.run(
                        "ss -tlnp 2>/dev/null | grep -q ':5173' && echo 'port_open' || echo 'port_closed'",
                        cwd="/home/user/react-app",
                    )
                    if "port_closed" in port_check.stdout:
                        print("Application checker: Port 5173 not listening — restarting Vite")
                        await sandbox.commands.run(
                            "nohup npm run dev -- --host 0.0.0.0 > /tmp/vite.log 2>&1 &",
                            cwd="/home/user/react-app",
                        )
                        await asyncio.sleep(10)  # Give Vite time to start
                    else:
                        print("Application checker: Port 5173 is already open")

                    vite_result = await sandbox.commands.run(
                        "curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://localhost:5173",
                        cwd="/home/user/react-app",
                    )
                    http_code = vite_result.stdout.strip()
                    if http_code == "200":
                        print("Application checker: Vite dev server is responding (HTTP 200)")
                    else:
                        print(f"Application checker: Vite dev server returned HTTP {http_code}")
                        runtime_errors.append(
                            {
                                "type": "vite_not_serving",
                                "error": (
                                    f"Vite dev server is not serving correctly (HTTP {http_code}). "
                                    "The app likely has a syntax error or crashes on startup. "
                                    "Check src/App.jsx and src/main.jsx for errors."
                                ),
                            }
                        )
                except Exception as e:
                    print(f"Application checker: Vite server check failed: {e}")
                    runtime_errors.append(
                        {
                            "type": "vite_check_failed",
                            "error": f"Could not reach Vite dev server: {str(e)}",
                        }
                    )

        except Exception as e:
            runtime_errors.append(
                {
                    "type": "file_check_failed",
                    "error": f"Failed to check application files: {str(e)}",
                }
            )

        update: dict = {
            "runtime_errors": runtime_errors,
            "current_node": "application_checker",
            "execution_log": [
                {
                    "node": "application_checker",
                    "status": "completed",
                    "runtime_errors": runtime_errors,
                }
            ],
        }

        if runtime_errors:
            current_retry = state.get("retry_count", {}).get("runtime_errors", 0)
            update["retry_count"] = {
                **state.get("retry_count", {}),
                "runtime_errors": current_retry + 1,
            }
            update["current_errors"] = {"runtime_errors": runtime_errors}
        else:
            update["success"] = True
            print("Application checker: No runtime errors found - setting success to True")

        if event_queue:
            safe_send_event(event_queue,
                {
                    "e": "app_check_complete",
                    "errors": runtime_errors,
                    "message": f"Application check completed. Found {len(runtime_errors)} runtime errors.",
                }
            )

        return update

    except Exception as e:
        error_msg = f"Application checker node error: {str(e)}"
        print(error_msg)

        if event_queue:
            safe_send_event(event_queue, {"e": "app_check_error", "message": error_msg})

        return {
            "current_node": "application_checker",
            "error_message": error_msg,
            "execution_log": [{"node": "application_checker", "status": "error", "error": error_msg}],
        }


def should_retry_builder_for_validation(state: GraphState) -> str:
    """Decide whether to retry builder for validation errors or continue"""
    validation_errors = state.get("validation_errors", [])
    retry_count = state.get("retry_count", {})
    max_retries = state.get("max_retries", 3)

    # Safety check: prevent infinite loops
    total_retries = sum(retry_count.values())
    if total_retries > 10:
        print(
            f"Maximum total retries reached ({total_retries}) - continuing to application checker"
        )
        return "application_checker"

    print(
        f"Code validator decision: {len(validation_errors)} errors, {retry_count.get('validation_errors', 0)} retries"
    )

    if not validation_errors:
        print("No validation errors - continuing to application checker")
        return "application_checker"

    current_retries = retry_count.get("validation_errors", 0)
    if current_retries < max_retries:
        print(
            f"Retrying builder for validation errors (attempt {current_retries + 1}/{max_retries})"
        )
        return "builder"
    else:
        print(
            f"Max retries reached for validation errors - continuing to application checker"
        )
        return "application_checker"


def should_retry_builder_or_finish(state: GraphState) -> str:
    """Decide whether to retry builder or finish based on runtime errors"""
    runtime_errors = state.get("runtime_errors", [])
    retry_count = state.get("retry_count", {})
    max_retries = state.get("max_retries", 3)

    # Safety check: prevent infinite loops
    total_retries = sum(retry_count.values())
    if total_retries > 10:
        print(f"Maximum total retries reached ({total_retries}) - forcing end")
        return "end"

    print(
        f"Application checker decision: {len(runtime_errors)} errors, {retry_count.get('runtime_errors', 0)} retries"
    )

    if not runtime_errors:
        print("No runtime errors - finishing successfully")
        return "end"

    current_retries = retry_count.get("runtime_errors", 0)
    if current_retries < max_retries:
        print(
            f"Retrying builder for runtime errors (attempt {current_retries + 1}/{max_retries})"
        )
        return "builder"
    else:
        print(f"Max retries reached for runtime errors - finishing with errors")
        return "end"
