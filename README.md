# Online Walkthrough API

A single FastAPI service that accepts raw walkthrough clip uploads and processes them into final concatenated videos. The service exposes dedicated routers for uploading and processing while sharing the same storage backend.

## Features

- `POST /api/upload` – multipart upload for video clips. Optional `filename` field lets you control saved names (e.g. `123_front`).
- `POST /api/process` – JSON request with `project_id` and `clips` list to concatenate clips using FFmpeg.
- Static delivery of uploaded clips at `/files/{filename}` (served from `uploads/raw/`).
- Static delivery of processed walkthroughs at `/processed/{filename}` (served from `uploads/final/`).
- CORS enabled for easy use from web or React Native clients.
- Health probe at `GET /health`.

## Project Layout

```
app/
  ├── main.py
  ├── paths.py
  ├── routers/
  │   ├── __init__.py
  │   ├── processing.py
  │   └── upload.py
  └── services/
      ├── __init__.py
      └── video_processor.py
Dockerfile
requirements.txt
uploads/
  ├── raw/
  └── final/
```

## Running Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Uploads are stored under `uploads/raw/`, and processed outputs are written to `uploads/final/`. When running outside Docker, ensure FFmpeg is installed and available on the PATH.

## Deploying with Coolify

1. Create a Coolify app pointing to this repository. Set the working directory to the repo root.
2. Use the provided `Dockerfile` build. It installs FFmpeg and Python dependencies automatically.
3. Expose port `8000` and map your domain (e.g. `/api/upload` and `/api/process`).
4. Attach a persistent volume mounted at `/app/uploads` so both raw and processed videos survive restarts.
5. Redeploy to apply configuration changes.

## Workflow

1. `POST /api/upload` each raw clip. The response includes the stored filename and a URL under `/files/`.
2. Once all clips for a project are uploaded, call `POST /api/process` with:

```json
{
  "project_id": "ABC123",
  "clips": ["ABC123_front.mp4", "ABC123_side.mp4"]
}
```

3. The response returns the processed filename and a `processed` URL ready for playback.

Because uploads and processing run in one service, there is no need for cross-service storage mounts—both routers operate on the same shared filesystem.
