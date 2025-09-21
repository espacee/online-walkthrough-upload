from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

import aiofiles
import re
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.paths import RAW_UPLOAD_DIR

router = APIRouter(tags=["uploads"])

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    """Persist the uploaded file to disk using async file IO."""
    async with aiofiles.open(destination, "wb") as out_file:
        while chunk := await upload_file.read(1024 * 1024):  # 1 MB chunks
            await out_file.write(chunk)


def build_target_filename(proposed_name: Optional[str], original_filename: str) -> str:
    """Return a safe filename, preserving extension and preventing traversal."""
    extension = Path(original_filename).suffix or ".mp4"
    extension = extension.lower()

    if proposed_name:
        cleaned = _UNSAFE_FILENAME_CHARS.sub("_", proposed_name.strip())
        cleaned = cleaned.strip("._") or "video"
        base_name = Path(cleaned).name
        if not Path(base_name).suffix:
            return f"{base_name}{extension}"
        return f"{Path(base_name).stem}{extension}"

    return f"{uuid4().hex}{extension}"


@router.post("/upload")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    filename: Optional[str] = Form(None),
) -> Dict[str, str]:
    """Accept video uploads and return a URL where the file can be accessed."""
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video files are allowed.")

    RAW_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    target_name = build_target_filename(filename, file.filename)
    file_path = RAW_UPLOAD_DIR / target_name

    await save_upload_file(file, file_path)

    file_url = request.url_for("raw-files", path=target_name)
    return {"filename": target_name, "url": str(file_url)}


@router.delete("/upload/{filename}")
async def delete_video(filename: str) -> Dict[str, str]:
    """Delete a previously uploaded raw clip."""
    safe_name = Path(filename).name
    file_path = RAW_UPLOAD_DIR / safe_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    file_path.unlink()
    return {"status": "deleted", "filename": safe_name}
