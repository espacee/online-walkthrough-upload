from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from video_processor import VideoProcessor

UPLOAD_ROOT = Path("uploads").resolve()
RAW_UPLOAD_DIR = UPLOAD_ROOT / "raw"
FINAL_UPLOAD_DIR = UPLOAD_ROOT / "final"

for directory in (RAW_UPLOAD_DIR, FINAL_UPLOAD_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Video Processing API",
    description="Concatenate raw clips into final walkthrough videos.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/processed/static",
    StaticFiles(directory=FINAL_UPLOAD_DIR, check_dir=False),
    name="processed-static",
)


class ProcessRequest(BaseModel):
    project_id: str = Field(..., min_length=1)
    clips: List[str] = Field(..., min_items=1, description="List of raw clip filenames")


@app.post("/api/process")
async def process_video(request: Request, payload: ProcessRequest) -> Dict[str, str]:
    processor = VideoProcessor(payload.project_id, RAW_UPLOAD_DIR, FINAL_UPLOAD_DIR)

    try:
        output_path = processor.concatenate_clips(payload.clips)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    output_name = Path(output_path).name
    output_url = request.url_for("serve_processed_video", filename=output_name)
    return {"output_url": str(output_url), "filename": output_name}


@app.get("/processed/{filename}")
async def serve_processed_video(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    file_path = FINAL_UPLOAD_DIR / safe_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Processed file not found.")

    return FileResponse(file_path)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}
