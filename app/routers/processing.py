import logging
import time
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.paths import FINAL_UPLOAD_DIR, RAW_UPLOAD_DIR
from app.services.video_processor import VideoProcessor

router = APIRouter(tags=["processing"])
logger = logging.getLogger(__name__)


class ProcessRequest(BaseModel):
    project_id: str = Field(..., min_length=1)
    clips: List[str] = Field(..., min_items=1, description="List of raw clip filenames")


@router.post("/process-walkthrough")
async def process_video(request: Request, payload: ProcessRequest) -> Dict[str, str]:
    request_id = request.headers.get("x-request-id") or uuid4().hex
    client_host = request.client.host if request.client else "unknown"
    clip_count = len(payload.clips)
    logger.info(
        "process-walkthrough start | request_id=%s project_id=%s client=%s clip_count=%d",
        request_id,
        payload.project_id,
        client_host,
        clip_count,
    )
    logger.debug(
        "process-walkthrough clips | request_id=%s project_id=%s clips=%s",
        request_id,
        payload.project_id,
        payload.clips,
    )

    processor = VideoProcessor(payload.project_id, RAW_UPLOAD_DIR, FINAL_UPLOAD_DIR)

    try:
        start_time = time.perf_counter()
        output_path = processor.process_walkthrough(payload.clips)
        duration_ms = (time.perf_counter() - start_time) * 1000
    except FileNotFoundError as exc:
        logger.error(
            "process-walkthrough missing clip | request_id=%s project_id=%s error=%s",
            request_id,
            payload.project_id,
            exc,
            exc_info=exc,
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        logger.error(
            "process-walkthrough validation error | request_id=%s project_id=%s error=%s",
            request_id,
            payload.project_id,
            exc,
            exc_info=exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error(
            "process-walkthrough ffmpeg failure | request_id=%s project_id=%s error=%s",
            request_id,
            payload.project_id,
            exc,
            exc_info=exc,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    output_name = Path(output_path).name
    output_url = request.url_for("processed-files", path=output_name)
    logger.info(
        "process-walkthrough complete | request_id=%s project_id=%s output=%s duration_ms=%.2f",
        request_id,
        payload.project_id,
        output_name,
        duration_ms,
    )
    return {
        "project_id": payload.project_id,
        "filename": output_name,
        "output_url": str(output_url),
    }
