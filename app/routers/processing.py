from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.paths import FINAL_UPLOAD_DIR, RAW_UPLOAD_DIR
from app.services.video_processor import VideoProcessor

router = APIRouter(tags=["processing"])


class ProcessRequest(BaseModel):
    project_id: str = Field(..., min_length=1)
    clips: List[str] = Field(..., min_items=1, description="List of raw clip filenames")


@router.post("/process")
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
    output_url = request.url_for("processed-files", path=output_name)
    return {
        "project_id": payload.project_id,
        "filename": output_name,
        "output_url": str(output_url),
    }
