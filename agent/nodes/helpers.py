"""
Shared helpers used by all graph nodes.

Provides: safe_send_event, store_message, _progress_ticker, _stream_agent_events, PlanSchema, NodeTimer.
"""

import asyncio
import time
import uuid
from typing import List

from pydantic import BaseModel, Field
from db.base import AsyncSessionLocal
from db.models import Message


# ─────────────────────────────────────────────────────────────
# Pydantic schema for planner structured output
# ─────────────────────────────────────────────────────────────

class PlanSchema(BaseModel):
    """Validated structure for the planner's output."""
    overview: str = Field(description="Application overview and purpose")
    components: List[str] = Field(default_factory=list, description="React components to create")
    pages: List[str] = Field(default_factory=list, description="Pages / routes")
    dependencies: List[str] = Field(default_factory=list, description="npm packages to install")
    file_structure: List[str] = Field(default_factory=list, description="File paths to create")
    implementation_steps: List[str] = Field(default_factory=list, description="Ordered build steps")


# ─────────────────────────────────────────────────────────────
# Node-level telemetry
# ─────────────────────────────────────────────────────────────

class NodeTimer:
    """Lightweight timer for measuring node execution time.

    Usage:
        timer = NodeTimer("planner")
        ...do work...
        log_entry = timer.stop()   # {"node": "planner", "duration_s": 4.2, ...}
    """

    def __init__(self, node_name: str):
        self.node_name = node_name
        self.start_time = time.time()

    def stop(self, status: str = "completed", **extra) -> dict:
        elapsed = round(time.time() - self.start_time, 2)
        entry = {
            "node": self.node_name,
            "status": status,
            "duration_s": elapsed,
            **extra,
        }
        print(f"⏱  {self.node_name} finished in {elapsed}s ({status})")
        return entry


# ─────────────────────────────────────────────────────────────
# Event helpers
# ─────────────────────────────────────────────────────────────

def safe_send_event(event_queue: asyncio.Queue, data: dict):
    """Helper to safely send events to queue."""
    if event_queue:
        try:
            event_queue.put_nowait(data)
        except Exception as e:
            print(f"Event queue send failed: {e}")


async def store_message(chat_id: str, role: str, content: str, event_type: str = None, tool_calls: list = None):
    """Helper to store a message in the database."""
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


# ─────────────────────────────────────────────────────────────
# Progress ticker + agent event streaming
# ─────────────────────────────────────────────────────────────

async def _progress_ticker(event_queue: asyncio.Queue, stop_event: asyncio.Event):
    """Send progress pulses every 8 seconds while the agent is generating code."""
    progress_messages = [
        "Writing components...",
        "Generating styles...",
        "Building pages...",
        "Creating utilities...",
        "Assembling application...",
        "Finalizing code...",
    ]
    tick = 0
    while not stop_event.is_set():
        await asyncio.sleep(8)
        if stop_event.is_set():
            break
        msg_index = min(tick, len(progress_messages) - 1)
        safe_send_event(event_queue, {"e": "progress", "message": progress_messages[msg_index]})
        tick += 1


async def stream_agent_events(
    agent_executor,
    messages: list,
    config: dict,
    event_queue: asyncio.Queue,
    project_id: str,
) -> None:
    """Stream agent events to the SSE queue.

    Tool calls are collected in-memory and flushed as a single DB row at the end.
    A background ticker sends progress pulses every 8s so the frontend stays alive.
    """

    thinking_sent = False
    tool_log: list[dict] = []

    # Background progress ticker
    stop_ticker = asyncio.Event()
    ticker_task = asyncio.create_task(_progress_ticker(event_queue, stop_ticker))

    try:
        async for event in agent_executor.astream_events(
            {"messages": messages}, version="v2", config=config
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                if not thinking_sent:
                    content = event["data"]["chunk"].content
                    if content:
                        if isinstance(content, list):
                            text_parts = [
                                (b if isinstance(b, str) else b.get("text", "") if isinstance(b, dict) else getattr(b, "text", ""))
                                for b in content
                            ]
                            content = "".join(filter(None, text_parts))
                        if content:
                            safe_send_event(event_queue, {"e": "thinking", "message": content})
                            thinking_sent = True
                continue

            if kind == "on_tool_start":
                stop_ticker.set()
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

                # Restart ticker for next tool call generation gap
                stop_ticker = asyncio.Event()
                ticker_task = asyncio.create_task(_progress_ticker(event_queue, stop_ticker))

    finally:
        stop_ticker.set()
        ticker_task.cancel()
        try:
            await ticker_task
        except asyncio.CancelledError:
            pass

    # Flush all tool calls as ONE DB row
    if tool_log:
        await store_message(
            chat_id=project_id,
            role="assistant",
            content=f"Executed {len(tool_log)} tool calls: {', '.join(t['name'] for t in tool_log)}",
            event_type="tool_summary",
            tool_calls=tool_log,
        )
