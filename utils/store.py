import os
import json
import time
from typing import List, Dict, Any


# Local filesystem storage - persists across sandbox restarts
PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..", "projects")


def get_store_path(id: str, filename: str):
    """Get local filesystem store path"""
    project_path = os.path.join(PROJECT_DIR, id)
    os.makedirs(project_path, exist_ok=True)
    return os.path.join(project_path, filename)


def save_json_store(id: str, filename: str, data: dict | list):
    """Save data to local filesystem."""
    try:
        file_path = get_store_path(id, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {filename} to local filesystem for project {id}")
    except Exception as e:
        print(f"Error saving {filename} for project {id}: {e}")


def load_json_store(id: str, filename: str):
    """Load data from local filesystem"""
    try:
        file_path = get_store_path(id, filename)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {filename} for project {id}: {e}")

    return {} if filename.endswith(".json") else []



# File persistence functions
def save_file_content(project_id: str, file_path: str, content: str):
    """Save file content to local filesystem"""
    try:
        sanitized_filename = file_path.replace("/", "_")
        file_store_path = get_store_path(project_id, sanitized_filename)
        with open(file_store_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved file {file_path} to local filesystem")
        return file_store_path
    except Exception as e:
        print(f"Error saving file {file_path} for project {project_id}: {e}")
        return ""



def load_file_content(project_id: str, file_path: str) -> str:
    """Load file content from local filesystem"""
    try:
        sanitized_filename = file_path.replace("/", "_")
        file_store_path = get_store_path(project_id, sanitized_filename)
        if os.path.exists(file_store_path):
            with open(file_store_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        print(f"Error loading file {file_path} for project {project_id}: {e}")
    return ""


def save_project_metadata(project_id: str, files: List[str], timestamp: float = None):
    """Save project metadata to local filesystem"""
    if timestamp is None:
        timestamp = time.time()

    metadata = {"project_id": project_id, "files": files, "timestamp": timestamp}

    save_json_store(project_id, "metadata.json", metadata)


def load_project_metadata(project_id: str) -> Dict[str, Any]:
    """Load project metadata from local filesystem"""
    return load_json_store(project_id, "metadata.json")


def get_stored_files(project_id: str) -> List[str]:
    """Get list of stored files for a project"""
    metadata = load_project_metadata(project_id)
    return metadata.get("files", [])


def file_exists_in_store(project_id: str, file_path: str) -> bool:
    """Check if a file exists in the local filesystem store"""
    try:
        sanitized_filename = file_path.replace("/", "_")
        file_store_path = get_store_path(project_id, sanitized_filename)
        return os.path.exists(file_store_path)
    except Exception as e:
        print(f"Error checking file {file_path} for project {project_id}: {e}")
        return False


def delete_stored_file(project_id: str, file_path: str):
    """Delete a file from the local filesystem store"""
    try:
        sanitized_filename = file_path.replace("/", "_")
        file_store_path = get_store_path(project_id, sanitized_filename)
        if os.path.exists(file_store_path):
            os.remove(file_store_path)
            print(f"Deleted file {file_path} from local filesystem")
    except Exception as e:
        print(f"Error deleting file {file_path} for project {project_id}: {e}")


def cleanup_project_store(project_id: str):
    """Clean up all stored files for a project from local filesystem"""
    try:
        project_path = os.path.join(PROJECT_DIR, project_id)
        if os.path.exists(project_path):
            import shutil

            shutil.rmtree(project_path)
            print(f"Cleaned up project store for {project_id}")
    except Exception as e:
        print(f"Error cleaning up project store for {project_id}: {e}")