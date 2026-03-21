"""
Cloudflare R2 storage client (S3-compatible).

Used to persist project file snapshots so the VM stays stateless.
All project files are stored under: r2://buildable-projects/{project_id}/{file_path}
"""

import os
import json
import boto3
from botocore.config import Config
from typing import Dict, List, Optional


def _get_client():
    """Create an S3 client pointed at Cloudflare R2."""
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("R2_ENDPOINT"),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("R2_SECRET_KEY"),
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 2, "mode": "standard"},
        ),
        region_name="auto",  # R2 uses "auto" as region
    )


BUCKET = os.getenv("R2_BUCKET_NAME", "buildable-projects")


def _is_configured() -> bool:
    """Check if R2 credentials are set."""
    return bool(os.getenv("R2_ACCESS_KEY") and os.getenv("R2_SECRET_KEY") and os.getenv("R2_ENDPOINT"))


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_file(project_id: str, file_path: str, content: str) -> bool:
    """Upload a single file to R2.

    Stored at: {project_id}/{file_path}
    e.g. "abc-123/src/App.jsx"
    """
    if not _is_configured():
        return False
    try:
        client = _get_client()
        key = f"{project_id}/{file_path}"
        client.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType=_guess_content_type(file_path),
        )
        return True
    except Exception as e:
        print(f"R2 upload error ({file_path}): {e}")
        return False


def upload_project_files(project_id: str, files: Dict[str, str]) -> int:
    """Upload multiple files to R2.

    Args:
        project_id: Project UUID
        files: Dict of {file_path: content}

    Returns:
        Number of files successfully uploaded.
    """
    if not _is_configured():
        print("R2 not configured — skipping upload")
        return 0

    uploaded = 0
    client = _get_client()
    for file_path, content in files.items():
        try:
            key = f"{project_id}/{file_path}"
            client.put_object(
                Bucket=BUCKET,
                Key=key,
                Body=content.encode("utf-8"),
                ContentType=_guess_content_type(file_path),
            )
            uploaded += 1
        except Exception as e:
            print(f"R2 upload error ({file_path}): {e}")

    print(f"R2: Uploaded {uploaded}/{len(files)} files for project {project_id}")
    return uploaded


def upload_json(project_id: str, filename: str, data: dict | list) -> bool:
    """Upload a JSON file (metadata, context) to R2."""
    if not _is_configured():
        return False
    try:
        client = _get_client()
        key = f"{project_id}/{filename}"
        client.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
        return True
    except Exception as e:
        print(f"R2 upload JSON error ({filename}): {e}")
        return False


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(project_id: str, file_path: str) -> Optional[str]:
    """Download a single file from R2."""
    if not _is_configured():
        return None
    try:
        client = _get_client()
        key = f"{project_id}/{file_path}"
        response = client.get_object(Bucket=BUCKET, Key=key)
        return response["Body"].read().decode("utf-8")
    except client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        print(f"R2 download error ({file_path}): {e}")
        return None


def download_project_files(project_id: str) -> Dict[str, str]:
    """Download all files for a project from R2.

    Returns:
        Dict of {file_path: content}
    """
    if not _is_configured():
        return {}

    files = {}
    try:
        client = _get_client()
        prefix = f"{project_id}/"
        paginator = client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                file_path = key[len(prefix):]  # Strip project_id prefix

                # Skip metadata/context files — they're loaded separately
                if file_path in ("metadata.json", "context.json"):
                    continue

                try:
                    response = client.get_object(Bucket=BUCKET, Key=key)
                    content = response["Body"].read().decode("utf-8")
                    files[file_path] = content
                except Exception as e:
                    print(f"R2 download error ({file_path}): {e}")

    except Exception as e:
        print(f"R2 list error for project {project_id}: {e}")

    print(f"R2: Downloaded {len(files)} files for project {project_id}")
    return files


def download_json(project_id: str, filename: str) -> Optional[dict]:
    """Download a JSON file from R2."""
    if not _is_configured():
        return None
    try:
        client = _get_client()
        key = f"{project_id}/{filename}"
        response = client.get_object(Bucket=BUCKET, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except Exception:
        # Not an error — file might not exist yet
        return None


# ---------------------------------------------------------------------------
# List / Delete
# ---------------------------------------------------------------------------

def list_project_files(project_id: str) -> List[str]:
    """List all file paths for a project in R2."""
    if not _is_configured():
        return []
    try:
        client = _get_client()
        prefix = f"{project_id}/"
        files = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                file_path = obj["Key"][len(prefix):]
                if file_path not in ("metadata.json", "context.json"):
                    files.append(file_path)
        return files
    except Exception as e:
        print(f"R2 list error for project {project_id}: {e}")
        return []


def delete_project(project_id: str) -> bool:
    """Delete all files for a project from R2."""
    if not _is_configured():
        return False
    try:
        client = _get_client()
        prefix = f"{project_id}/"
        paginator = client.get_paginator("list_objects_v2")

        objects_to_delete = []
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                objects_to_delete.append({"Key": obj["Key"]})

        if objects_to_delete:
            # Delete in batches of 1000 (S3 limit)
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i:i + 1000]
                client.delete_objects(
                    Bucket=BUCKET,
                    Delete={"Objects": batch},
                )

        print(f"R2: Deleted {len(objects_to_delete)} objects for project {project_id}")
        return True
    except Exception as e:
        print(f"R2 delete error for project {project_id}: {e}")
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _guess_content_type(file_path: str) -> str:
    """Guess content type from file extension."""
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    return {
        "js": "application/javascript",
        "jsx": "application/javascript",
        "ts": "application/typescript",
        "tsx": "application/typescript",
        "css": "text/css",
        "html": "text/html",
        "json": "application/json",
        "svg": "image/svg+xml",
        "md": "text/markdown",
    }.get(ext, "text/plain")
