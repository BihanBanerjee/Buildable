from typing import TypedDict, List, Dict, Any, Optional
from e2b_code_interpreter import AsyncSandbox
import asyncio


class GraphState(TypedDict):
    """State schema for the LangGraph multi-agent system"""

    project_id: str  # Also serves as chat_id
    user_prompt: str
    enhanced_prompt: str

    # Planning phase
    plan: Optional[Dict[str, Any]]

    # Building phase
    files_created: List[str]
    files_modified: List[str]

    # Error tracking
    current_errors: Dict[str, Any]
    validation_errors: List[Dict[str, Any]]
    runtime_errors: List[Dict[str, Any]]

    # Retry tracking
    retry_count: Dict[str, int]
    max_retries: int

    # Environment
    sandbox: Optional[AsyncSandbox]
    event_queue: Optional[asyncio.Queue]

    # Node execution tracking
    current_node: str
    execution_log: List[Dict[str, Any]]

    # Results
    success: bool
    error_message: Optional[str]