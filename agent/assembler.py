"""
Project assembler — merges LLM-generated files with the BASE_TEMPLATE.

The LLM generates only feature files (components, pages, context).
This module overlays them onto the immutable base template to produce
the complete project file set.
"""

import copy
import re

from .base_template import BASE_TEMPLATE


def assemble_project(generated_files: list[dict]) -> dict[str, str]:
    """Merge generated files with BASE_TEMPLATE.

    1. Start with a copy of BASE_TEMPLATE
    2. Overlay generated files
    3. Strip 'import ./App.css' lines (Tailwind-only)
    4. Force vite.config.js from template (never let LLM override)
    5. Return flat {path: content} dict

    Args:
        generated_files: List of {path: str, content: str} from the build agent.

    Returns:
        Complete project as {relative_path: file_content}.
    """
    project = copy.deepcopy(BASE_TEMPLATE)

    for f in generated_files:
        path = f.get("path", "")
        content = f.get("content", "")

        # Strip App.css imports (Tailwind-only system)
        content = re.sub(r"import\s+[\"']\.\/App\.css[\"'];?\n?", "", content)

        project[path] = content

    # Always force the base-template vite.config.js so allowedHosts / host
    # settings required by the E2B sandbox are never lost.
    project["vite.config.js"] = BASE_TEMPLATE["vite.config.js"]

    return project
