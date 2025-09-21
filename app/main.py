from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.paths import FINAL_UPLOAD_DIR, RAW_UPLOAD_DIR, ensure_directories
from app.routers import processing, upload

ensure_directories()

app = FastAPI(
    title="Online Walkthrough API",
    description="Upload raw clips and process them into final walkthrough videos.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api")
app.include_router(processing.router, prefix="/api")

app.mount("/files", StaticFiles(directory=RAW_UPLOAD_DIR, check_dir=False), name="raw-files")
app.mount(
    "/processed",
    StaticFiles(directory=FINAL_UPLOAD_DIR, check_dir=False),
    name="processed-files",
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
