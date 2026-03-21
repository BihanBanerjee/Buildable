import os
import json
import time
from typing import List, Dict, Any

from utils import r2


# Local filesystem storage - used as cache, R2 is the source of truth
PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "projects")


def get_store_path(id: str, filename: str):
    """Get local filesystem store path"""
    project_path = os.path.join(PROJECT_DIR, id)
    os.makedirs(project_path, exist_ok=True)
    return os.path.join(project_path, filename)


def save_json_store(id: str, filename: str, data: dict | list):
    """Save JSON data to R2 (primary) and local filesystem (cache)."""
    # R2 — primary storage
    r2.upload_json(id, filename, data)

    # Local — cache
    try:
        file_path = get_store_path(id, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {filename} for project {id}")
    except Exception as e:
        print(f"Error saving {filename} locally for project {id}: {e}")


def load_json_store(id: str, filename: str):
    """Load JSON data from local cache first, then R2 fallback."""
    # Try local cache first
    try:
        file_path = get_store_path(id, filename)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {filename} locally for project {id}: {e}")

    # Fallback to R2
    data = r2.download_json(id, filename)
    if data is not None:
        print(f"Loaded {filename} from R2 for project {id}")
        return data

    return {} if filename.endswith(".json") else []


# ---------------------------------------------------------------------------
# File persistence — R2 primary, local cache
# ---------------------------------------------------------------------------

def save_file_content(project_id: str, file_path: str, content: str):
    """Save file content to R2 (primary) and local filesystem (cache)."""
    # R2 — primary (uses original file path, not sanitized)
    r2.upload_file(project_id, file_path, content)

    # Local — cache (sanitized filename for flat directory)
    try:
        sanitized_filename = file_path.replace("/", "_")
        file_store_path = get_store_path(project_id, sanitized_filename)
        with open(file_store_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_store_path
    except Exception as e:
        print(f"Error saving file {file_path} locally for project {project_id}: {e}")
        return ""


def save_project_files_bulk(project_id: str, files: Dict[str, str]):
    """Save multiple files to R2 in bulk + local cache.

    Args:
        files: Dict of {file_path: content} e.g. {"src/App.jsx": "import..."}
    """
    # R2 — bulk upload
    r2.upload_project_files(project_id, files)

    # Local — cache each file
    for file_path, content in files.items():
        try:
            sanitized_filename = file_path.replace("/", "_")
            file_store_path = get_store_path(project_id, sanitized_filename)
            with open(file_store_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"Error caching file {file_path} locally: {e}")


def load_file_content(project_id: str, file_path: str) -> str:
    """Load file content from local cache first, then R2 fallback."""
    # Try local cache first
    try:
        sanitized_filename = file_path.replace("/", "_")
        file_store_path = get_store_path(project_id, sanitized_filename)
        if os.path.exists(file_store_path):
            with open(file_store_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        print(f"Error loading file {file_path} locally: {e}")

    # Fallback to R2
    content = r2.download_file(project_id, file_path)
    if content is not None:
        print(f"Loaded {file_path} from R2 for project {project_id}")
        return content

    return ""


def load_all_project_files(project_id: str) -> Dict[str, str]:
    """Load all project files from R2.

    Returns:
        Dict of {file_path: content}
    """
    return r2.download_project_files(project_id)


def save_project_metadata(project_id: str, files: List[str], timestamp: float = None):
    """Save project metadata to R2 + local cache."""
    if timestamp is None:
        timestamp = time.time()

    metadata = {"project_id": project_id, "files": files, "timestamp": timestamp}
    save_json_store(project_id, "metadata.json", metadata)


def load_project_metadata(project_id: str) -> Dict[str, Any]:
    """Load project metadata from local cache or R2."""
    return load_json_store(project_id, "metadata.json")


def get_stored_files(project_id: str) -> List[str]:
    """Get list of stored files for a project."""
    metadata = load_project_metadata(project_id)
    return metadata.get("files", [])


def file_exists_in_store(project_id: str, file_path: str) -> bool:
    """Check if a file exists in local cache."""
    try:
        sanitized_filename = file_path.replace("/", "_")
        file_store_path = get_store_path(project_id, sanitized_filename)
        return os.path.exists(file_store_path)
    except Exception as e:
        print(f"Error checking file {file_path} for project {project_id}: {e}")
        return False


def delete_stored_file(project_id: str, file_path: str):
    """Delete a file from local cache."""
    try:
        sanitized_filename = file_path.replace("/", "_")
        file_store_path = get_store_path(project_id, sanitized_filename)
        if os.path.exists(file_store_path):
            os.remove(file_store_path)
    except Exception as e:
        print(f"Error deleting file {file_path} for project {project_id}: {e}")


def cleanup_project_store(project_id: str):
    """Clean up all stored files for a project from R2 and local cache."""
    # R2
    r2.delete_project(project_id)

    # Local cache
    try:
        project_path = os.path.join(PROJECT_DIR, project_id)
        if os.path.exists(project_path):
            import shutil
            shutil.rmtree(project_path)
            print(f"Cleaned up project store for {project_id}")
    except Exception as e:
        print(f"Error cleaning up project store for {project_id}: {e}")
