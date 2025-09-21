from pathlib import Path

UPLOAD_ROOT = Path("uploads")
RAW_UPLOAD_DIR = UPLOAD_ROOT / "raw"
FINAL_UPLOAD_DIR = UPLOAD_ROOT / "final"


def ensure_directories() -> None:
    for directory in (RAW_UPLOAD_DIR, FINAL_UPLOAD_DIR):
        directory.mkdir(parents=True, exist_ok=True)
