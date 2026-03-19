from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator


class GraphState(TypedDict):
    """State schema for the LangGraph multi-agent system"""

    project_id: str  # Also serves as chat_id
    user_prompt: str
    enhanced_prompt: str
    is_first_message: bool  # True for initial build, False for follow-up prompts

    # Planning phase
    plan: Optional[Dict[str, Any]]

    # Building phase
    # Annotated with operator.add so lists accumulate across builder retries
    # rather than being overwritten. Deduplicate at read time in service.py.
    files_created: Annotated[List[str], operator.add]
    files_modified: List[str]

    # Error tracking
    current_errors: Dict[str, Any]
    validation_errors: List[Dict[str, Any]]
    runtime_errors: List[Dict[str, Any]]

    # Retry tracking
    retry_count: Dict[str, int]
    max_retries: int

    # Node execution tracking
    current_node: str
    execution_log: Annotated[List[Dict[str, Any]], operator.add]

    # Full OpenRouter model ID for the builder node (e.g. "google/gemini-2.5-pro")
    builder_model: str

    # Results
    success: bool
    error_message: Optional[str]
