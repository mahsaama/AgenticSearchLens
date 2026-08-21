"""Local API server behind the "Explore Your Own Traces" section of the
project's GitHub Pages site (agentic-search-atlas.html).

Runs entirely on your machine. Uploaded exports and any extracted stats
never leave localhost — the browser page just calls this server over
http://127.0.0.1. See README.md for the full walkthrough.

Usage:
    python -m src.local_server            # serves on http://127.0.0.1:8420

Endpoints:
    GET  /health                          liveness check for the page's
                                           "connected?" indicator
    POST /upload  (multipart form-data:   saves an export to
                   platform, file)        data/<platform>/user_0/conversations.json
    POST /analyze (json: {"platforms":    runs personal_explorer.explore_platform
                   [...]})                for each platform, returns per-platform
                                           stats/errors
"""

import json
import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.personal_explorer import PLATFORMS, NoDataError, explore_platform

load_dotenv()

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

DATA_DIR = "data"
HOST = os.getenv("LOCAL_SERVER_HOST")
PORT = int(os.getenv("LOCAL_SERVER_PORT"))

app = FastAPI(title="Agentic Search Atlas — Local Explorer")

# This server only ever binds to localhost, so the only thing CORS gates is
# which *browser tabs* on your own machine may call it — not who else on the
# network can reach it. Left permissive by default; set
# LOCAL_SERVER_ALLOWED_ORIGIN in .env (e.g. to your GitHub Pages URL) to
# restrict it.
_allowed_origin = os.getenv("LOCAL_SERVER_ALLOWED_ORIGIN", "").strip()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_allowed_origin] if _allowed_origin else ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    platforms: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(platform: str, file: UploadFile):
    if platform not in PLATFORMS:
        raise HTTPException(400, f"Unknown platform: {platform!r}. Use one of {PLATFORMS}.")

    raw = await file.read()
    try:
        json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"'{file.filename}' is not valid JSON: {exc}") from exc

    dest_dir = f"{DATA_DIR}/{platform}/user_0"
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = f"{dest_dir}/conversations.json"
    with open(dest_path, "wb") as f:
        f.write(raw)

    logger.info("Saved %s upload (%d bytes) to %s", platform, len(raw), dest_path)
    return {"platform": platform, "saved_to": dest_path, "bytes": len(raw)}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    unknown = [p for p in req.platforms if p not in PLATFORMS]
    if unknown:
        raise HTTPException(400, f"Unknown platform(s): {unknown}. Use any of {PLATFORMS}.")

    results = {}
    errors = {}
    for platform in req.platforms:
        try:
            results[platform] = explore_platform(platform, base_dir=DATA_DIR)
        except NoDataError as exc:
            errors[platform] = str(exc)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the page, don't 500 the batch
            logger.exception("Analysis failed for platform=%s", platform)
            errors[platform] = f"{type(exc).__name__}: {exc}"

    return {"results": results, "errors": errors}


if __name__ == "__main__":
    print(f"Agentic Search Atlas local explorer running at http://{HOST}:{PORT}")
    print("Keep this running, then use the 'Explore Your Own Traces' section on the page.")
    uvicorn.run(app, host=HOST, port=PORT)
