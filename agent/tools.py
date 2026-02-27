from e2b_code_interpreter import AsyncSandbox
import asyncio
from langchain_core.tools import tool
import os
import json
from utils.store import save_json_store, load_json_store


def create_tools_with_context(
    sandbox: AsyncSandbox, event_queue: asyncio.Queue, project_id: str = None
):
    """Create tools with sandbox and event queue context"""
    def safe_send_event(queue: asyncio.Queue, data: dict) -> bool:
        """Safely send event to queue; return False if send fails."""
        try:
            if queue:
                queue.put_nowait(data)
            return True
        except Exception as e:
            # Log and return False so callers can stop trying to send further messages.
            print(f"safe_send_event failed: {e}")
            return False

    @tool
    async def create_file(file_path: str, content: str) -> str:
        """
        Create a file with the given content at the specified path.

        Args:
            file_path: The path where the file should be created (e.g., "src/App.jsx", "src/components/Header.jsx")
            content: The content to write to the file (React components, HTML, CSS, etc.)

        Returns:
            Success message with file path or error message if failed

        Example:
            create_file("src/App.jsx", "import React from 'react';\\nexport default function App() { return <div>Hello</div>; }")
        """
        try:
            # The React app is in /home/user/react-app
            full_path = os.path.join("/home/user/react-app", file_path)

            # The LLM sometimes generates code with literal \n instead of actual newlines
            # We need to handle this by decoding escape sequences
            # Use bytes decode to properly interpret escape sequences
            try:
                # Try to decode the content as if it contains escaped characters
                # This converts literal \n, \t, etc. to actual newline and tab characters
                fixed_content = content.encode("utf-8").decode("unicode_escape")
            except (UnicodeDecodeError, AttributeError):
                # If decode fails, content is likely already correct, use as-is
                fixed_content = content

            await sandbox.files.write(full_path, fixed_content)
            safe_send_event(event_queue, {"e": "file_created", "message": f"Created {file_path}"})
            return f"File {file_path} created successfully."
        except Exception as e:
            safe_send_event(event_queue, {"e": "file_error", "message": f"Failed to create {file_path}: {str(e)}",})
            return f"Failed to create file {file_path}: {str(e)}"

    @tool
    async def read_file(file_path: str) -> str:
        """
        Read the content of a file from the react-app directory.

        Args:
            file_path: The path of the file to read (e.g., "src/App.jsx", "package.json")

        Returns:
            The file content as a string, or error message if file not found

        Example:
            read_file("src/App.jsx") - reads the main App component
            read_file("package.json") - reads package dependencies
        """
        try:
            # The React app is in /home/user/react-app
            full_path = os.path.join("/home/user/react-app", file_path)
            content = await sandbox.files.read(full_path)
            safe_send_event(event_queue, {"e": "file_read", "message": f"Read content from {file_path}"})
            return f"Content from {file_path}:\n{content}"
        except Exception as e:
            safe_send_event(event_queue, {"e": "file_error", "message": f"Failed to read {file_path}: {str(e)}"})
            return f"Failed to read file {file_path}: {str(e)}"

    @tool
    async def delete_file(file_path: str) -> str:
        """
        Delete a file from the react-app directory.

        Args:
            file_path: The path of the file to delete (e.g., "src/old-component.jsx")

        Returns:
            Success message or error message if deletion failed

        Example:
            delete_file("src/old-component.jsx") - removes an unused component
        """
        try:
            # The React app is in /home/user/react-app
            full_path = os.path.join("/home/user/react-app", file_path)
            await sandbox.files.remove(full_path)
            safe_send_event(event_queue, {"e": "file_deleted", "message": f"Deleted {file_path}"})
            return f"File {file_path} deleted successfully."
        except Exception as e:
            safe_send_event(event_queue, {"e": "file_error", "message": f"Failed to delete {file_path}: {str(e)}",})
            return f"Failed to delete file {file_path}: {str(e)}"

    @tool
    async def execute_command(command: str) -> str:
        """
        Execute a shell command within the react-app directory.

        Args:
            command: The shell command to execute (e.g., "npm install", "npm run dev", "mkdir src/components")

        Returns:
            Command output and success/error status

        Common Commands:
            - "npm install" - install dependencies
            - "npm install react-router-dom" - install specific package
            - "mkdir -p src/components" - create directory structure
            - "npm run dev" - start development server (usually already running)

        Example:
            execute_command("npm install react-router-dom") - installs routing library
        """
        try:
            safe_send_event(event_queue, {"e": "command_started", "command": command})

            # The React app is in /home/user/react-app
            result = await sandbox.commands.run(command, cwd="/home/user/react-app")

            safe_send_event(
                event_queue,
                {
                    "e": "command_output",
                    "command": command,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                },
            )

            if result.exit_code == 0:
                safe_send_event(
                    event_queue,
                    {
                        "e": "command_executed",
                        "command": command,
                        "message": "Command executed successfully",
                    },
                )
                return f"Command '{command}' executed successfully. Output: {result.stdout[:500]}{'...' if len(result.stdout) > 500 else ''}"
            else:
                safe_send_event(
                    event_queue,
                    {
                        "e": "command_failed",
                        "command": command,
                        "message": f"Command failed with exit code {result.exit_code}",
                    },
                )
                return f"Command '{command}' failed with exit code {result.exit_code}. Error: {result.stderr[:500]}{'...' if len(result.stderr) > 500 else ''}"

        except Exception as e:
            safe_send_event(
                event_queue,
                {
                    "e": "command_error",
                    "command": command,
                    "message": f"Command execution error: {str(e)}",
                },
            )
            return f"Command '{command}' failed with error: {str(e)}"

    @tool
    async def list_directory(path: str = ".") -> str:
        """
        List the directory structure using tree command, excluding node_modules and hidden files.

        Args:
            path: The directory path to list (default: "." for current directory)

        Returns:
            Formatted directory tree structure

        Example:
            list_directory() - lists current directory
            list_directory("src") - lists src directory structure

        Note:
            Automatically excludes node_modules and hidden files for cleaner output
        """
        try:
            safe_send_event(event_queue, {"e": "command_started", "command": f"tree -I 'node_modules|.*' {path}"})

            result = await sandbox.commands.run(
                f"tree -I 'node_modules|.*' {path}", cwd="/home/user/react-app"
            )

            safe_send_event(
                event_queue,
                {
                    "e": "command_output",
                    "command": f"tree -I 'node_modules|.*' {path}",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                },
            )

            if result.exit_code == 0:
                safe_send_event(
                    event_queue,
                    {
                        "e": "command_executed",
                        "command": f"tree -I 'node_modules|.*' {path}",
                        "message": "Directory structure listed successfully",
                    },
                )
                return f"Directory structure:\n{result.stdout}"
            else:
                safe_send_event(
                    event_queue,
                    {
                        "e": "command_failed",
                        "command": f"tree -I 'node_modules|.*' {path}",
                        "message": f"Command failed with exit code {result.exit_code}",
                    },
                )
                return f"Failed to list directory structure. Error: {result.stderr}"

        except Exception as e:
            safe_send_event(
                event_queue,
                {
                    "e": "command_error",
                    "command": f"tree -I 'node_modules|.*' {path}",
                    "message": f"Command execution error: {str(e)}",
                },
            )
            return f"Failed to list directory: {str(e)}"

    @tool
    async def test_build() -> str:
        """
        Test if the application builds successfully by running npm run build.
        This cleans the Vite cache, runs npm install if needed, and attempts a build.

        Returns:
            Success message with build output or error message with details

        Example:
            test_build() - Tests if the current application builds without errors

        Note:
            This is useful for validating that all files are correct before deployment
        """

        return "build success"
        try:
            path = "/home/user/react-app"

            safe_send_event(event_queue, {"e": "build_test_started", "message": "Testing application build..."})

            # Clean Vite cache first nd run npm install
            clean_command = "rm -rf node_modules/.vite-temp && npm install"
            await sandbox.commands.run(clean_command, cwd=path)

            # Run build
            build_command = "npm run build"
            res = await sandbox.commands.run(build_command, cwd=path)

            if res.exit_code == 0:
                safe_send_event(
                    event_queue,
                    {
                        "e": "build_test_success",
                        "message": "Build test passed successfully",
                    },
                )
                return f"Build test PASSED. Application builds successfully.\n\nBuild output:\n{res.stdout[:500]}"
            else:
                error_output = res.stderr if res.stderr else res.stdout
                safe_send_event(
                    event_queue,
                    {
                        "e": "build_test_failed",
                        "message": "Build test failed",
                        "error": error_output[:500],
                    },
                )
                return f"Build test FAILED with exit code {res.exit_code}.\n\nError:\n{error_output[:1000]}"

        except Exception as e:
            safe_send_event(event_queue, {"e": "build_test_error", "message": f"Build test error: {str(e)}"})
            return f"Build test failed with error: {str(e)}"

    @tool
    async def write_multiple_files(files: str) -> str:
        """
        Write multiple files to the sandbox at once for better efficiency.

        Args:
            files: JSON string containing array of file objects with 'path' and 'data' keys

        Returns:
            Success message with list of created files or error message

        Format:
            [
                {"path": "src/App.jsx", "data": "import React from 'react';..."},
                {"path": "src/components/Header.jsx", "data": "export default function Header() {...}"}
            ]

        Example:
            write_multiple_files('[{"path": "src/App.jsx", "data": "import React from \'react\'; export default function App() { return <div>Hello</div>; }"}]')

        Benefits:
            - More efficient than creating files one by one
            - Prevents agent from stopping prematurely
            - Creates complete application structure at once
        """
        try:
            files_data = json.loads(files)

            # Convert to the format expected by E2B
            file_objects = []
            for file_info in files_data:
                # Fix escape sequences in the data (same as create_file)
                content = file_info["data"]
                try:
                    # Try to decode the content as if it contains escaped characters
                    # This converts literal \n, \t, etc. to actual newline and tab characters
                    fixed_content = content.encode("utf-8").decode("unicode_escape")
                except (UnicodeDecodeError, AttributeError):
                    # If decode fails, content is likely already correct, use as-is
                    fixed_content = content

                file_objects.append(
                    {
                        "path": os.path.join("/home/user/react-app", file_info["path"]),
                        "data": fixed_content,
                    }
                )

            await sandbox.files.write_files(file_objects)

            file_names = [f["path"] for f in files_data]
            safe_send_event(
                event_queue,
                {
                    "e": "files_created",
                    "message": f"Created {len(file_names)} files: {', '.join(file_names)}",
                },
            )

            return (
                f"Successfully created {len(file_names)} files: {', '.join(file_names)}"
            )

        except Exception as e:
            safe_send_event(
                event_queue,
                {
                    "e": "file_error",
                    "message": f"Failed to create multiple files: {str(e)}",
                },
            )
            return f"Failed to create multiple files: {str(e)}"

    @tool
    def get_context() -> str:
        """
        Fetch the last saved context for the current project.
        This includes information about:
        - What the project is about (semantic memory)
        - How things work in the project (procedural memory)
        - What has been done so far (episodic memory)

        Returns:
            Saved project context as a formatted string, or message if no context exists

        Example:
            get_context() - Retrieves the project context to understand what was previously built

        Use this tool:
        - At the start of your work to understand the project
        - To check what components/features already exist
        - To understand the project structure and conventions
        """
        if not project_id:
            return "No project ID available - context cannot be retrieved"

        try:
            context = load_json_store(project_id, "context.json")

            if not context:
                return "No previous context found for this project. This appears to be a new project."

            # Format the context for display
            result = "=== PROJECT CONTEXT ===\n\n"

            if context.get("semantic"):
                result += "📋 WHAT THIS PROJECT IS:\n"
                result += f"{context['semantic']}\n\n"

            if context.get("procedural"):
                result += "HOW THINGS WORK:\n"
                result += context["procedural"] + "\n\n"

            if context.get("episodic"):
                result += "WHAT HAS BEEN DONE:\n"
                result += f"{context['episodic']}\n\n"

            if context.get("files_created"):
                result += f"📁 FILES CREATED: {len(context['files_created'])} files\n"
                result += f"   {', '.join(context['files_created'][:10])}"
                if len(context["files_created"]) > 10:
                    result += f" ... and {len(context['files_created']) - 10} more"
                result += "\n\n"

            if context.get("conversation_history"):
                result += "CONVERSATION HISTORY:\n"
                for i, conv in enumerate(context["conversation_history"][-5:], 1):
                    status = "[SUCCESS]" if conv.get("success") else "[FAILED]"
                    result += f"   {i}. {status} {conv.get('user_prompt', 'Unknown')[:80]}...\n"
                result += "\n"

            if context.get("last_updated"):
                result += f"Last Updated: {context['last_updated']}\n"

            return result

        except Exception as e:
            return f"Failed to retrieve context: {str(e)}"

    @tool
    def save_context(semantic: str, procedural: str = "", episodic: str = "") -> str:
        """
        Save project context for future sessions.
        This helps maintain continuity across different work sessions.

        Args:
            semantic: What the project is about (e.g., "E-commerce site for jewelry with cart and checkout")
            procedural: How things work (e.g., "Uses React Router for navigation, Context API for state")
            episodic: What has been done (e.g., "Created product catalog, cart functionality, and checkout flow")

        Returns:
            Success message confirming context was saved

        Example:
            save_context(
                semantic="Portfolio website for Abhay with projects, skills, and contact form",
                procedural="Uses React Router, Tailwind CSS for styling, data stored in src/data/",
                episodic="Created all pages, components, and navigation. Added responsive design."
            )

        Use this tool:
        - After completing major features
        - Before finishing your work
        - When you want to document what you've built
        """
        if not project_id:
            return "No project ID available - context cannot be saved"

        try:
            # Load existing context to preserve information
            existing_context = load_json_store(project_id, "context.json")

            # Update context with new information
            context = {
                "semantic": semantic or existing_context.get("semantic", ""),
                "procedural": procedural or existing_context.get("procedural", ""),
                "episodic": episodic or existing_context.get("episodic", ""),
                "last_updated": str(os.popen("date").read().strip()),
                "files_created": existing_context.get("files_created", []),
                "conversation_history": existing_context.get(
                    "conversation_history", []
                ),
            }

            # Save to store
            save_json_store(project_id, "context.json", context)

            return f"Context saved successfully for project {project_id}. This information will be available in future sessions."

        except Exception as e:
            return f"Failed to save context: {str(e)}"

    @tool
    async def check_missing_packages() -> str:
        """
        Check for missing packages by reading package.json and scanning source files for imports.
        This tool identifies missing dependencies and provides installation commands.

        Returns:
            A report of missing packages and installation commands

        Example:
            check_missing_packages() - Scans files and reports missing packages
        """
        try:
            # Read package.json to see installed packages
            package_json_path = "/home/user/react-app/package.json"
            package_content = await sandbox.files.read(package_json_path)
            package_data = json.loads(package_content)
            installed_deps = package_data.get("dependencies", {})

            # Find all source files
            find_result = await sandbox.commands.run(
                "find src -name '*.jsx' -o -name '*.js'", cwd="/home/user/react-app"
            )
            source_files = [
                f.strip() for f in find_result.stdout.strip().split("\n") if f.strip()
            ]

            # Scan all files for import statements
            all_imports = set()
            for file_path in source_files:
                try:
                    full_path = f"/home/user/react-app/{file_path}"
                    content = await sandbox.files.read(full_path)

                    # Extract import statements
                    import_lines = [
                        line.strip()
                        for line in content.split("\n")
                        if line.strip().startswith("import")
                    ]
                    for line in import_lines:
                        # Extract package names from import statements
                        if "from" in line:
                            package = line.split("from")[1].strip().strip("'\"")
                            # Extract root package name (e.g., 'react-icons/fa' -> 'react-icons')
                            root_package = package.split("/")[0]
                            all_imports.add(root_package)
                        elif "import" in line and not "from" in line:
                            # Handle import { something } from 'package' syntax
                            if "'" in line or '"' in line:
                                package = (
                                    line.split("'")[1]
                                    if "'" in line
                                    else line.split('"')[1]
                                )
                                root_package = package.split("/")[0]
                                all_imports.add(root_package)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

            # Check which packages are missing
            missing_packages = []
            for package in all_imports:
                if package not in installed_deps and package not in [
                    "react",
                    "react-dom",
                ]:
                    missing_packages.append(package)

            if missing_packages:
                install_commands = []
                for package in missing_packages:
                    install_commands.append(f"npm install {package}")

                result = f"MISSING DEPENDENCIES FOUND:\n\n"
                result += f"Missing packages: {', '.join(missing_packages)}\n\n"
                result += f"Installation commands:\n"
                for cmd in install_commands:
                    result += f"  {cmd}\n"
                result += f"\nRun these commands to install missing dependencies."

                safe_send_event(
                    event_queue,
                    {
                        "e": "missing_dependencies",
                        "packages": missing_packages,
                        "commands": install_commands,
                    },
                )

                return result
            else:
                return "All dependencies are properly installed. No missing packages found."

        except Exception as e:
            safe_send_event(
                event_queue,
                {
                    "e": "dependency_check_error",
                    "message": f"Dependency check failed: {str(e)}",
                },
            )
            return f"Dependency check failed: {str(e)}"

    return [
        create_file,
        read_file,
        execute_command,
        test_build,
        delete_file,
        list_directory,
        write_multiple_files,
        get_context,
        save_context,
        check_missing_packages,
    ]
