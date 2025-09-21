from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Dict
from uuid import uuid4
import aiofiles

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


async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    """Persist the uploaded file to disk using async file IO."""
    async with aiofiles.open(destination, "wb") as out_file:
        while chunk := await upload_file.read(1024 * 1024):  # 1 MB chunks
            await out_file.write(chunk)


@app.post("/api/upload")
async def upload_video(request: Request, file: UploadFile = File(...)) -> Dict[str, str]:
    """Accept video uploads and return a URL where the file can be accessed."""
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video files are allowed.")

    original_extension = Path(file.filename).suffix or ".mp4"
    safe_extension = original_extension.lower()
    unique_name = f"{uuid4().hex}{safe_extension}"
    file_path = UPLOAD_DIR / unique_name

    await save_upload_file(file, file_path)

    file_url = request.url_for("serve_video", filename=unique_name)
    return {"url": str(file_url)}


@app.get("/files/{filename}")
async def serve_video(filename: str):
    """Serve uploaded video files by filename."""
    safe_name = Path(filename).name  # Prevent path traversal
    file_path = UPLOAD_DIR / safe_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(file_path)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint for monitoring."""
    return {"status": "ok"}
