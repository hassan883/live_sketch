"""
FastAPI wrapper for live_sketch animate_svg.py

Exposes async job management for SVG sketch animation.
Jobs run as subprocesses with progress parsing from stdout.
"""

import os
import re
import uuid
import shutil
import signal
import asyncio
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Live Sketch API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths — works inside Docker (/app) or locally
LIVE_SKETCH_DIR = Path(__file__).parent
ANIMATE_SCRIPT = LIVE_SKETCH_DIR / "animate_svg.py"
OUTPUT_DIR = LIVE_SKETCH_DIR / "job_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


class AnimateRequest(BaseModel):
    svg: str
    caption: str
    num_frames: int = 24
    num_iter: int = 200
    guidance_scale: float = 30.0


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    progress: int = 0  # 0-100
    error: Optional[str] = None


# In-memory job store
jobs: Dict[str, dict] = {}


@app.post("/api/animate")
async def submit_animation(request: AnimateRequest) -> dict:
    if not ANIMATE_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"animate_svg.py not found at {ANIMATE_SCRIPT}",
        )

    job_id = str(uuid.uuid4())
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True)

    # Write SVG to temp file
    svg_path = job_dir / "input.svg"
    svg_path.write_text(request.svg)

    # Also write a scaled version (live_sketch expects target without .svg extension)
    svg_input_dir = job_dir / "svg_input"
    svg_input_dir.mkdir()
    scaled_svg = svg_input_dir / "input_scaled1.svg"
    scaled_svg.write_text(request.svg)

    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "error": None,
        "process": None,
        "job_dir": str(job_dir),
        "num_iter": request.num_iter,
    }

    # Launch subprocess
    asyncio.create_task(_run_animation(job_id, request))

    return {"jobId": job_id}


async def _run_animation(job_id: str, request: AnimateRequest) -> None:
    job = jobs.get(job_id)
    if not job:
        return

    job_dir = Path(job["job_dir"])
    svg_target = str(job_dir / "svg_input" / "input_scaled1")
    output_folder = str(job_dir / "output")

    cmd = [
        "python", "-u",
        str(ANIMATE_SCRIPT),
        "--target", svg_target,
        "--caption", request.caption,
        "--output_folder", output_folder,
        "--num_frames", str(request.num_frames),
        "--num_iter", str(request.num_iter),
        "--guidance_scale", str(request.guidance_scale),
        "--save_vid_iter", "100",
        "--optim_points", "True",
        "--opt_points_with_mlp", "True",
        "--lr_local", "0.005",
        "--predict_global_frame_deltas", "1",
        "--split_global_loss", "True",
        "--guidance_scale_global", "40",
        "-augment_frames", "True",
    ]

    try:
        job["status"] = "running"
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(LIVE_SKETCH_DIR),
            env=env,
        )
        job["process"] = process

        # Parse stdout for progress — match both "iteration 100" and tqdm "100/1001"
        iteration_pattern = re.compile(r"iter[ation]*\s*[:=]?\s*(\d+)", re.IGNORECASE)
        tqdm_pattern = re.compile(r"\s(\d+)/(\d+)\s")
        num_iter = request.num_iter

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").strip()
            if decoded:
                print(f"[{job_id[:8]}] {decoded}")
                match = iteration_pattern.search(decoded)
                if match:
                    current_iter = int(match.group(1))
                    job["progress"] = min(int(current_iter / num_iter * 100), 99)
                else:
                    match = tqdm_pattern.search(decoded)
                    if match:
                        current_iter = int(match.group(1))
                        total_iter = int(match.group(2))
                        job["progress"] = min(int(current_iter / total_iter * 100), 99)

        await process.wait()

        if process.returncode == 0:
            gif_path = _find_result_gif(job_dir)
            if gif_path:
                job["status"] = "completed"
                job["progress"] = 100
                job["result_path"] = str(gif_path)
            else:
                job["status"] = "failed"
                job["error"] = "Animation completed but no output GIF found"
        else:
            job["status"] = "failed"
            job["error"] = f"Process exited with code {process.returncode}"

    except asyncio.CancelledError:
        if job.get("process"):
            job["process"].send_signal(signal.SIGTERM)
        job["status"] = "failed"
        job["error"] = "Job cancelled"
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
    finally:
        job["process"] = None


def _find_result_gif(job_dir: Path) -> Optional[Path]:
    """Search for the output GIF in the job directory."""
    for pattern in ["**/*HQ_gif.gif", "**/*.gif", "**/*.mp4"]:
        matches = list(job_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


@app.get("/api/animate/{job_id}")
async def get_job_status(job_id: str) -> JobStatus:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        error=job.get("error"),
    )


@app.get("/api/animate/{job_id}/result")
async def get_job_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    result_path = job.get("result_path")
    if not result_path or not Path(result_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    suffix = Path(result_path).suffix.lower()
    media_type = "image/gif" if suffix == ".gif" else "video/mp4"
    return FileResponse(result_path, media_type=media_type)


@app.delete("/api/animate/{job_id}")
async def cancel_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    process = job.get("process")
    if process and process.returncode is None:
        process.send_signal(signal.SIGTERM)

    job_dir = Path(job["job_dir"])
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)

    del jobs[job_id]
    return {"status": "cancelled"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
