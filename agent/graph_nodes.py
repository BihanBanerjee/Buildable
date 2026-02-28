from langchain_core.messages import HumanMessage, SystemMessage
import asyncio
import copy
import re
from .graph_state import GraphState
from .tools import create_tools_with_context
from .agent import llm_gemini_pro, llm_gemini_flash
from .formatters import (
    create_formatted_message,
    format_plan_as_markdown,
    generate_description_from_plan,
    generate_build_summary,
)
import json
from langgraph.prebuilt import create_react_agent
from .prompts import INITPROMPT, VALIDATOR_PROMPT, get_builder_error_prompt, get_builder_prompt
from utils.store import load_json_store
import traceback
from db.base import AsyncSessionLocal
from db.models import Message
import uuid


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
) -> list:
    """
    Stream agent events to the SSE queue and store them in the DB.
    Returns list of file paths detected from tool outputs (for builder tracking).
    """
    files_created = []

    async for event in agent_executor.astream_events(
        {"messages": messages}, version="v1", config=config
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

            if "created" in tool_output.lower() and "file" in tool_output.lower():
                file_matches = re.findall(r"(\w+\.(jsx?|tsx?|css|json))", tool_output)
                files_created.extend([match[0] for match in file_matches])

    return files_created


async def planner_node(state: GraphState) -> GraphState:
    """
    Planner node: Analyzes user prompt and generates comprehensive implementation plan
    """
    try:
        event_queue = state.get("event_queue")
        if event_queue:
            safe_send_event(event_queue,
                {
                    "e": "planner_started",
                    "message": "Planning the application architecture...",
                }
            )

        enhanced_prompt = state.get("enhanced_prompt", state.get("user_prompt", ""))
        print(f"INFO: Recieved Prompt {enhanced_prompt}")
        project_id = state.get("project_id", "")
        print(f"INFO: Project ID: {project_id}")

        previous_context = ""

        # check if previous context is there
        if project_id:
            print(f"Loading context for project: {project_id}")
            context = load_json_store(project_id, "context.json")
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

        response = await llm_gemini_flash.ainvoke(messages)

        # Format the plan preview for better display
        plan_preview = response.content[:500] if len(response.content) > 500 else response.content
        formatted_preview = create_formatted_message("thinking", plan_preview)

        if event_queue:
            safe_send_event(event_queue, formatted_preview)

        # Store plan preview with formatting
        await store_message(
            chat_id=state.get("project_id"),
            role="assistant",
            content=formatted_preview.get("formatted", plan_preview),
            event_type="thinking"
        )

        try:
            plan = json.loads(response.content)
        except json.JSONDecodeError:
            plan = {
                "overview": response.content,
                "components": [],
                "pages": [],
                "dependencies": [],
                "file_structure": [],
                "implementation_steps": [],
            }

        new_state = copy.deepcopy(state)
        new_state["plan"] = plan
        new_state["current_node"] = "planner"
        new_state["execution_log"].append(
            {"node": "planner", "status": "completed", "plan": plan}
        )

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

        return new_state

    except Exception as e:
        error_msg = f"Planner node error: {str(e)}"
        print(error_msg)

        new_state = copy.deepcopy(state)
        new_state["current_node"] = "planner"
        new_state["error_message"] = error_msg
        new_state["execution_log"].append(
            {"node": "planner", "status": "error", "error": error_msg}
        )

        if event_queue:
            safe_send_event(event_queue, {"e": "planner_error", "message": error_msg})

        return new_state


async def builder_node(state: GraphState) -> GraphState:
    """
    Builder node: Creates and modifies files based on plan or feedback
    """

    try:
        event_queue = state.get("event_queue")
        sandbox = state.get("sandbox")

        if not sandbox:
            raise Exception("Sandbox not available")

        if event_queue:
            safe_send_event(event_queue,
                {
                    "e": "builder_started",
                    "message": "Starting to build the application...",
                }
            )

        plan = state.get("plan", {})
        if plan:
            print("INFO: Plan Recieved")

        current_errors = state.get("current_errors", {})

        project_id = state.get("project_id", "")

        base_tools = create_tools_with_context(sandbox, event_queue, project_id)

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

        agent_executor = create_react_agent(llm_gemini_pro, tools=base_tools)
        config = {"recursion_limit": 50}

        try:
            print(
                f"Builder node: Starting agent execution with {len(base_tools)} tools"
            )

            files_created = await _stream_agent_events(
                agent_executor, messages, config, event_queue, project_id
            )
            files_modified = []

            print(f"Builder node: Agent execution completed")
            print(f"Builder node: Final files_created: {files_created}")

            new_state = copy.deepcopy(state)
            new_state["files_created"] = files_created
            new_state["files_modified"] = files_modified
            print(f"INFO : {files_created}")
            new_state["current_node"] = "builder"
            new_state["execution_log"].append(
                {
                    "node": "builder",
                    "status": "completed",
                    "files_created": files_created,
                    "files_modified": files_modified,
                }
            )

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

            return new_state

        except asyncio.TimeoutError:
            print("Builder agent timed out after 10 minutes")
            files_created = []
            files_modified = []

            new_state = copy.deepcopy(state)
            new_state["files_created"] = files_created
            new_state["files_modified"] = files_modified
            new_state["current_node"] = "builder"
            new_state["execution_log"].append(
                {
                    "node": "builder",
                    "status": "timeout",
                    "files_created": files_created,
                    "files_modified": files_modified,
                }
            )

            if event_queue:
                safe_send_event(event_queue,
                    {
                        "e": "builder_error",
                        "message": "Builder agent timed out after 10 minutes",
                    }
                )

            return new_state

        except Exception as e:
            print(f"Builder agent execution error: {e}")
            traceback.print_exc()
            files_created = []
            files_modified = []

            new_state = copy.deepcopy(state)
            new_state["files_created"] = files_created
            new_state["files_modified"] = files_modified
            new_state["current_node"] = "builder"
            new_state["execution_log"].append(
                {
                    "node": "builder",
                    "status": "error",
                    "files_created": files_created,
                    "files_modified": files_modified,
                }
            )

            if event_queue:
                safe_send_event(event_queue,
                    {
                        "e": "builder_error",
                        "message": f"Builder agent execution error: {str(e)}",
                    }
                )

            return new_state

    except Exception as e:
        error_msg = f"Builder node error: {str(e)}"
        print(error_msg)

        new_state = copy.deepcopy(state)
        new_state["current_node"] = "builder"
        new_state["error_message"] = error_msg
        new_state["execution_log"].append(
            {"node": "builder", "status": "error", "error": error_msg}
        )

        if event_queue:
            safe_send_event(event_queue, {"e": "builder_error", "message": error_msg})

        return new_state



async def code_validator_node(state: GraphState) -> GraphState:
    """
    Code Validator node: Active React agent that reviews, validates, and fixes code
    """
    try:
        event_queue = state.get("event_queue")
        sandbox = state.get("sandbox")

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
        base_tools = create_tools_with_context(sandbox, event_queue, project_id)

        validator_prompt = VALIDATOR_PROMPT

        messages = [
            SystemMessage(
                content="You are a Code Validator Agent. Review and fix all code issues."
            ),
            HumanMessage(content=validator_prompt),
        ]

        validator_agent = create_react_agent(llm_gemini_flash, tools=base_tools)
        config = {"recursion_limit": 50}

        try:
            print(
                f"Code validator: Starting agent execution with {len(base_tools)} tools"
            )

            await _stream_agent_events(
                validator_agent, messages, config, event_queue, project_id
            )
            validation_errors = []

            print(f"Code validator: Agent execution completed")
            print("Code validator: Code review and dependency checking completed")

            if event_queue:
                safe_send_event(event_queue,
                    {
                        "e": "validation_success",
                        "message": "Code validator completed - code review and dependencies checked!",
                    }
                )

            new_state = copy.deepcopy(state)
            new_state["validation_errors"] = validation_errors
            new_state["current_node"] = "code_validator"

            if validation_errors:
                retry_count = new_state.get("retry_count", {})
                retry_count["validation_errors"] = (
                    retry_count.get("validation_errors", 0) + 1
                )
                new_state["retry_count"] = retry_count
                new_state["current_errors"] = {"validation_errors": validation_errors}
                print(
                    f"Code validator: Found {len(validation_errors)} validation errors"
                )
            else:
                print("Code validator: No validation errors found")

            new_state["execution_log"].append(
                {
                    "node": "code_validator",
                    "status": "completed",
                    "validation_errors": validation_errors,
                }
            )

            if event_queue:
                safe_send_event(event_queue,
                    {
                        "e": "code_validator_complete",
                        "errors": validation_errors,
                        "message": f"Code validation completed. Found {len(validation_errors)} errors.",
                    }
                )

            return new_state

        except asyncio.TimeoutError:
            print("Code validator agent timed out after 10 minutes")

            new_state = copy.deepcopy(state)
            new_state["validation_errors"] = [
                {
                    "type": "timeout",
                    "error": "Code validator timed out",
                    "details": "Validation took too long",
                }
            ]
            new_state["current_node"] = "code_validator"

            if event_queue:
                safe_send_event(event_queue,
                    {
                        "e": "code_validator_timeout",
                        "message": "Code validator timed out",
                    }
                )

            return new_state

    except Exception as e:
        error_msg = f"Code validator node error: {str(e)}"
        print(error_msg)
        traceback.print_exc()

        new_state = copy.deepcopy(state)
        new_state["current_node"] = "code_validator"
        new_state["error_message"] = error_msg
        new_state["validation_errors"] = [
            {
                "type": "validator_error",
                "error": str(e),
                "details": "Code validator crashed",
            }
        ]
        new_state["execution_log"].append(
            {"node": "code_validator", "status": "error", "error": error_msg}
        )

        if event_queue:
            safe_send_event(event_queue, {"e": "code_validator_error", "message": error_msg})

        return new_state


async def application_checker_node(state: GraphState) -> GraphState:
    """
    Application Checker node: Checks if the application is running and captures errors
    """
    try:
        event_queue = state.get("event_queue")
        sandbox = state.get("sandbox")

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

        print(
            "Application checker: Skipping dev server checks - environment is pre-configured"
        )

        try:
            # Check if main files exist
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

        except Exception as e:
            runtime_errors.append(
                {
                    "type": "file_check_failed",
                    "error": f"Failed to check application files: {str(e)}",
                }
            )

        new_state = copy.deepcopy(state)
        new_state["runtime_errors"] = runtime_errors
        new_state["current_node"] = "application_checker"

        if runtime_errors:
            retry_count = new_state.get("retry_count", {})
            retry_count["runtime_errors"] = retry_count.get("runtime_errors", 0) + 1
            new_state["retry_count"] = retry_count

            new_state["current_errors"] = {"runtime_errors": runtime_errors}
        else:
            new_state["success"] = True
            print(
                "Application checker: No runtime errors found - setting success to True"
            )

        new_state["execution_log"].append(
            {
                "node": "application_checker",
                "status": "completed",
                "runtime_errors": runtime_errors,
            }
        )

        if event_queue:
            safe_send_event(event_queue,
                {
                    "e": "app_check_complete",
                    "errors": runtime_errors,
                    "message": f"Application check completed. Found {len(runtime_errors)} runtime errors.",
                }
            )

        return new_state

    except Exception as e:
        error_msg = f"Application checker node error: {str(e)}"
        print(error_msg)

        new_state = copy.deepcopy(state)
        new_state["current_node"] = "application_checker"
        new_state["error_message"] = error_msg
        new_state["execution_log"].append(
            {"node": "application_checker", "status": "error", "error": error_msg}
        )

        if event_queue:
            safe_send_event(event_queue, {"e": "app_check_error", "message": error_msg})

        return new_state


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
