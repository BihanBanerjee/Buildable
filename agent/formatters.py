"""
Message formatting utilities for better frontend display
"""
import json
import re
from typing import List


def format_json_as_markdown(data: dict, title: str = None) -> str:
    """
    Format JSON data as readable markdown
    """
    lines = []
    
    if title:
        lines.append(f"## {title}\n")
    
    # Handle plan structure
    if "planTitle" in data or "applicationOverview" in data:
        return format_plan_as_markdown(data)
    
    # Generic JSON formatting
    for key, value in data.items():
        formatted_key = key.replace("_", " ").title()
        
        if isinstance(value, dict):
            lines.append(f"### {formatted_key}\n")
            lines.append(format_dict_section(value, indent=0))
        elif isinstance(value, list):
            lines.append(f"### {formatted_key}\n")
            lines.append(format_list_section(value, indent=0))
        else:
            lines.append(f"**{formatted_key}:** {value}\n")
    
    return "\n".join(lines)


def format_plan_as_markdown(plan: dict) -> str:
    """
    Format a detailed implementation plan as markdown
    """
    lines = []
    
    # Handle nested plan structure (e.g., {"plan": {...}})
    if "plan" in plan and isinstance(plan["plan"], dict):
        plan = plan["plan"]
    
    # Title
    if "planTitle" in plan:
        lines.append(f"# {plan['planTitle']}\n")
    elif "name" in plan.get("applicationOverviewAndPurpose", {}):
        lines.append(f"# {plan['applicationOverviewAndPurpose']['name']}\n")
    
    # Overview - handle both old and new format
    overview = plan.get("applicationOverview") or plan.get("applicationOverviewAndPurpose")
    if overview:
        lines.append("## Application Overview\n")
        if isinstance(overview, dict):
            if "title" in overview or "name" in overview:
                title = overview.get("title") or overview.get("name")
                lines.append(f"**Name:** {title}\n")
            if "purpose" in overview:
                lines.append(f"**Purpose:** {overview['purpose']}\n")
            if "features" in overview or "coreFeatures" in overview:
                features = overview.get("features") or overview.get("coreFeatures", [])
                if isinstance(features, list) and features:
                    lines.append("\n**Features:**")
                    for feature in features:
                        lines.append(f"- {feature}")
                    lines.append("")
            if "targetAudience" in overview:
                lines.append(f"**Target Audience:** {overview['targetAudience']}\n")
        else:
            lines.append(f"{overview}\n")
    
    # Component Hierarchy - handle both formats
    hierarchy = plan.get("componentHierarchy") or plan.get("componentHierarchyAndStructure")
    if hierarchy:
        lines.append("## Component Hierarchy\n")
        if isinstance(hierarchy, dict):
            if "rootComponent" in hierarchy:
                lines.append(f"**Root:** `{hierarchy['rootComponent']}`\n")
            if "structure" in hierarchy and isinstance(hierarchy["structure"], list):
                for component in hierarchy["structure"]:
                    if isinstance(component, dict):
                        name = component.get("name", "Unknown")
                        desc = component.get("description", "")
                        lines.append(f"### `{name}`")
                        lines.append(f"{desc}\n")
                        
                        if "children" in component:
                            lines.append("**Children:**")
                            for child in component["children"]:
                                lines.append(f"- `{child}`")
                            lines.append("")
        elif isinstance(hierarchy, list):
            # New format: direct list of components
            for component in hierarchy:
                if isinstance(component, dict):
                    name = component.get("name", "Unknown")
                    desc = component.get("description", "")
                    lines.append(f"### `{name}`")
                    if desc:
                        lines.append(f"{desc}\n")
                    
                    # Props
                    if "props" in component:
                        props = component["props"]
                        if props and props != "None" and props != "None (consumes context)":
                            lines.append(f"**Props:** `{props}`\n")
                    
                    # State
                    if "state" in component:
                        state = component["state"]
                        if state and state != "None":
                            lines.append(f"**State:** `{state}`\n")
                    
                    # Children
                    if "children" in component:
                        children = component["children"]
                        if isinstance(children, list) and children:
                            lines.append("**Children:**")
                            for child in children:
                                lines.append(f"- `{child}`")
                            lines.append("")
    
    # Page Routing
    routing = plan.get("pageRoutingStructure")
    if routing:
        lines.append("## Page Routing\n")
        if isinstance(routing, dict):
            if "type" in routing:
                lines.append(f"**Type:** {routing['type']}\n")
            if "description" in routing:
                lines.append(f"{routing['description']}\n")
            if "routes" in routing and isinstance(routing["routes"], list):
                lines.append("**Routes:**")
                for route in routing["routes"]:
                    if isinstance(route, dict):
                        path = route.get("path", "/")
                        component = route.get("component", "")
                        description = route.get("description", "")
                        lines.append(f"- `{path}` → `{component}`")
                        if description:
                            lines.append(f"  - {description}")
                lines.append("")
    
    # Dependencies
    deps = plan.get("requiredDependencies")
    if deps:
        lines.append("## Required Dependencies\n")
        if isinstance(deps, dict):
            for category, packages in deps.items():
                category_name = category.replace("_", " ").title()
                lines.append(f"### {category_name}")
                if isinstance(packages, list):
                    for pkg in packages:
                        if isinstance(pkg, dict):
                            name = pkg.get("name", "")
                            version = pkg.get("version", "")
                            purpose = pkg.get("purpose", "")
                            install_cmd = pkg.get("installationCommand", "")
                            if version:
                                lines.append(f"- **{name}** `{version}` - {purpose}")
                            else:
                                lines.append(f"- **{name}** - {purpose}")
                            if install_cmd:
                                lines.append(f"  ```bash\n  {install_cmd}\n  ```")
                        else:
                            lines.append(f"- {pkg}")
                lines.append("")
        elif isinstance(deps, list):
            # New format: direct list
            for pkg in deps:
                if isinstance(pkg, dict):
                    name = pkg.get("name", "")
                    purpose = pkg.get("purpose", "")
                    install_cmd = pkg.get("installationCommand", "")
                    lines.append(f"- **{name}** - {purpose}")
                    if install_cmd:
                        lines.append(f"  ```bash\n  {install_cmd}\n  ```")
                else:
                    lines.append(f"- {pkg}")
            lines.append("")
    
    # File Structure
    file_structure = plan.get("fileStructure")
    if file_structure:
        lines.append("## File Structure\n")
        if isinstance(file_structure, dict):
            root = file_structure.get("root", "./")
            lines.append(f"**Root:** `{root}`\n")
            structure = file_structure.get("structure", [])
            if isinstance(structure, list):
                lines.append("```")
                for item in structure:
                    lines.append(item)
                lines.append("```\n")
    
    # Implementation Steps
    steps = plan.get("implementationSteps")
    if steps:
        lines.append("## Implementation Steps\n")
        if isinstance(steps, list):
            for i, step in enumerate(steps, 1):
                if isinstance(step, dict):
                    step_num = step.get("step", i)
                    title = step.get("title") or step.get("stage", f"Step {step_num}")
                    description = step.get("description", "")
                    tasks = step.get("tasks", [])
                    
                    lines.append(f"### Step {step_num}: {title}\n")
                    if description:
                        lines.append(f"{description}\n")
                    
                    if isinstance(tasks, list) and tasks:
                        lines.append("**Tasks:**")
                        for task in tasks:
                            if isinstance(task, str):
                                lines.append(f"- {task}")
                            elif isinstance(task, list):
                                for subtask in task:
                                    lines.append(f"  - {subtask}")
                    lines.append("")
    
    return "\n".join(lines)


def format_dict_section(data: dict, indent: int = 0) -> str:
    """Format dictionary as markdown with indentation"""
    lines = []
    prefix = "  " * indent
    
    for key, value in data.items():
        formatted_key = key.replace("_", " ").title()
        if isinstance(value, (dict, list)):
            lines.append(f"{prefix}**{formatted_key}:**")
            if isinstance(value, dict):
                lines.append(format_dict_section(value, indent + 1))
            else:
                lines.append(format_list_section(value, indent + 1))
        else:
            lines.append(f"{prefix}**{formatted_key}:** {value}")
    
    return "\n".join(lines)


def format_list_section(items: list, indent: int = 0) -> str:
    """Format list as markdown with indentation"""
    lines = []
    prefix = "  " * indent
    
    for item in items:
        if isinstance(item, dict):
            lines.append(format_dict_section(item, indent))
        elif isinstance(item, list):
            lines.append(format_list_section(item, indent + 1))
        else:
            lines.append(f"{prefix}- {item}")
    
    return "\n".join(lines)


def format_code_block(code: str, language: str = "") -> str:
    """Format code as markdown code block"""
    return f"```{language}\n{code}\n```"


def format_thinking_message(content: str) -> str:
    """
    Format AI thinking/reasoning messages for better readability
    """
    # Try to parse as JSON first
    try:
        data = json.loads(content)
        return format_json_as_markdown(data)
    except (json.JSONDecodeError, TypeError):
        pass
    
    # If not JSON, format as regular text with better structure
    # Remove excessive newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Add markdown formatting for common patterns
    # Convert "SECTION:" to headers
    content = re.sub(r'^([A-Z][A-Z\s]+):$', r'### \1', content, flags=re.MULTILINE)
    
    # Convert numbered lists
    content = re.sub(r'^\s*(\d+)\.\s+', r'\1. ', content, flags=re.MULTILINE)
    
    return content


def generate_description_from_plan(plan: dict, user_prompt: str) -> str:
    """
    Generate a human-readable description from the plan JSON.
    Produces Loveable-style intro text with features and design notes.
    """
    lines = []

    overview = (
        plan.get("applicationOverview")
        or plan.get("applicationOverviewAndPurpose")
        or plan.get("overview")
        or {}
    )

    if isinstance(overview, dict):
        purpose = overview.get("purpose") or overview.get("description") or ""
        title = overview.get("title") or overview.get("name") or ""

        if purpose:
            lines.append(purpose)
        elif title:
            lines.append(f"I'll build **{title}** for you.")

        features = overview.get("features") or overview.get("coreFeatures", [])
        if isinstance(features, list) and features:
            feature_str = ", ".join(str(f) for f in features[:8])
            lines.append(f"\n**Key Features:** {feature_str}.")

        design = overview.get("design") or overview.get("style") or overview.get("theme")
        if design:
            if isinstance(design, dict):
                design_str = ", ".join(f"{k}: {v}" for k, v in design.items() if v)
                lines.append(f"\n**Design:** {design_str}.")
            else:
                lines.append(f"\n**Design:** {design}.")

    elif isinstance(overview, str) and overview.strip():
        lines.append(overview[:500])

    if not lines:
        title = plan.get("planTitle") or plan.get("title") or ""
        if title:
            lines.append(f"I'll build **{title}** based on your requirements.")
        else:
            lines.append("I'll build your application based on your requirements.")

    return "\n".join(lines)


def generate_build_summary(files_created: list, files_modified: list, plan: dict) -> str:
    """
    Generate a human-readable completion summary after building.
    """
    lines = []

    overview = (
        plan.get("applicationOverview")
        or plan.get("applicationOverviewAndPurpose")
        or plan.get("overview")
        or {}
    )
    app_name = ""
    if isinstance(overview, dict):
        app_name = overview.get("title") or overview.get("name") or ""

    if app_name:
        lines.append(f"Your **{app_name}** is ready!")
    else:
        lines.append("Your application is ready!")

    total = len(files_created) + len(files_modified)
    if total > 0:
        parts = []
        if files_created:
            parts.append(f"created {len(files_created)} files")
        if files_modified:
            parts.append(f"modified {len(files_modified)} files")
        lines.append(f"I {' and '.join(parts)} to build your app.")

    if isinstance(overview, dict):
        features = overview.get("features") or overview.get("coreFeatures", [])
        if isinstance(features, list) and features:
            lines.append("\nIt includes:")
            for f in features[:6]:
                lines.append(f"- **{f}**")

    return "\n".join(lines)


def create_formatted_message(message_type: str, content: any, **extra_data) -> dict:
    """
    Create a formatted WebSocket message with both raw data and formatted display
    
    Args:
        message_type: Type of message (thinking, plan, code, etc.)
        content: The content to format (can be string, dict, etc.)
        **extra_data: Additional data to include in the message
    
    Returns:
        Dict with 'type', 'content', 'formatted', and any extra data
    """
    message = {
        "e": message_type,
        **extra_data
    }
    
    # Handle different content types
    if isinstance(content, dict):
        message["content"] = content
        if message_type in ["plan", "planner_complete"]:
            message["formatted"] = format_plan_as_markdown(content)
        else:
            message["formatted"] = format_json_as_markdown(content)
    elif isinstance(content, str):
        message["content"] = content
        if message_type == "thinking":
            message["formatted"] = format_thinking_message(content)
        else:
            message["formatted"] = content
    else:
        message["content"] = str(content)
        message["formatted"] = str(content)
    
    return message
