from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4
import aiofiles
import re

# Base directory for storing uploaded files
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Video Upload API",
    description="Simple FastAPI server for uploading and serving video files.",
)

# Allow React Native (and other clients) to call the API without CORS issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=UPLOAD_DIR, check_dir=False), name="uploaded-files")


async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    """Persist the uploaded file to disk using async file IO."""
    async with aiofiles.open(destination, "wb") as out_file:
        while chunk := await upload_file.read(1024 * 1024):  # 1 MB chunks
            await out_file.write(chunk)


_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def build_target_filename(proposed_name: Optional[str], original_filename: str) -> str:
    """Return a safe filename, preserving extension and preventing traversal."""
    extension = Path(original_filename).suffix or ".mp4"
    extension = extension.lower()

    if proposed_name:
        cleaned = _UNSAFE_FILENAME_CHARS.sub("_", proposed_name.strip())
        cleaned = cleaned.strip("._") or "video"
        # Ensure users cannot sneak path separators
        base_name = Path(cleaned).name
        if not Path(base_name).suffix:
            return f"{base_name}{extension}"
        return f"{Path(base_name).stem}{extension}"

    return f"{uuid4().hex}{extension}"


@app.post("/api/upload")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    filename: Optional[str] = Form(None),
) -> Dict[str, str]:
    """Accept video uploads and return a URL where the file can be accessed."""
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video files are allowed.")

    target_name = build_target_filename(filename, file.filename)
    file_path = UPLOAD_DIR / target_name

    await save_upload_file(file, file_path)

    file_url = request.url_for("uploaded-files", path=target_name)
    return {"url": str(file_url)}

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint for monitoring."""
    return {"status": "ok"}
