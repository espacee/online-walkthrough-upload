# Online Walkthrough Backend Monorepo

This repository contains the APIs behind the Online Walkthrough experience. It is organised as a monorepo with two FastAPI services that can be deployed independently (for example, as separate Coolify applications).

## Services

### `video-upload-api`
- Handles multipart video uploads at `POST /api/upload`.
- Validates that uploads are video files and stores them in `uploads/`.
- Optional `filename` form field lets clients persist clips using meaningful names (e.g. `123_front`).
- Serves files via `GET /files/{filename}` and static hosting mounted at `/files`.
- Health probe at `GET /health`.
- Docker image listens on port `8000`.

**Run locally**
```bash
cd video-upload-api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### `video-processing-api`
- Accepts JSON requests at `POST /api/process` containing a `project_id` and `clips` list with raw clip filenames.
- Uses FFmpeg (available in the container) to concatenate the clip sequence into `uploads/final/{project_id}_final.mp4`.
- Serves processed videos via `GET /processed/{filename}` and exposes the `uploads/final` directory at `/processed/static`.
- Health probe at `GET /health`.
- Docker image listens on port `8001`.

> **Note:** The service expects raw clips to be present in `uploads/raw/`. Ensure FFmpeg is available when running outside of Docker.

## Deploying with Coolify

Deploy each service as its own Coolify app:
1. Point the app to this repository and set the working directory to either `video-upload-api` or `video-processing-api`.
2. Choose the Dockerfile build strategy (Coolify will automatically use the Dockerfile inside that directory).
3. Expose the appropriate port (`8000` for uploads, `8001` for processing) and configure domains as needed.
4. Persist the `uploads/` directory (Coolify volume) if you need files to survive container redeployments.

## Repository Structure

```
video-upload-api/
  ├── Dockerfile
  ├── main.py
  ├── requirements.txt
  └── uploads/
video-processing-api/
  ├── Dockerfile
  ├── main.py
  ├── requirements.txt
  └── video_processor.py
README.md
```

Each service remains self-contained so you can continue iterating independently while keeping a single source of truth for the backend code.
