"""Self-hosted NFL model server: scheduler + API + dashboard.

Runs the same pipeline as the GitHub Actions, but on your own hardware
with cron-grade reliability, plus an API the dashboard (and future UI)
can call to trigger runs and read live status.

    uvicorn app:app --host 0.0.0.0 --port 8000   (from server/)

Jobs shell out to the existing nfl-model scripts — one code path,
whether GitHub or your Proxmox box runs it. If GIT_SYNC=1, each job
pulls before running and commits/pushes results after, so the repo on
GitHub stays the source of truth and your offsite backup.
"""
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO / "nfl-model"
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

JOBS = {
    "fetch":   {"cmd": ["python3", "props_fetch.py"],      "cron": "30 11 * * 2"},
    "improve": {"cmd": ["python3", "improve.py"],           "cron": "0 12 * * 2"},
    "ledger":  {"cmd": ["python3", "ledger.py"],            "cron": "10 12 * * 2"},
    "collect": {"cmd": ["python3", "props_collector.py"],   "cron": "0 13 * * 2,0"},
    "export":  {"cmd": ["python3", "export_dashboard.py"],  "cron": "20 12 * * 2"},
}
# More aggressive schedules? Edit the cron strings above (min hour dom mon dow)
# and restart. e.g. daily improve: "0 12 * * *". Your hardware, your rules.

status: dict[str, dict] = {name: {} for name in JOBS}


def git(*args) -> str:
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def run_job(name: str):
    job = JOBS[name]
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log_file = LOG_DIR / f"{name}.log"

    if os.environ.get("GIT_SYNC") == "1":
        git("pull", "--rebase", "origin", "main")

    r = subprocess.run(job["cmd"], cwd=MODEL_DIR, capture_output=True,
                       text=True, timeout=3600)
    log_file.write_text(f"=== {started} exit={r.returncode}\n{r.stdout}\n{r.stderr}")

    if os.environ.get("GIT_SYNC") == "1" and r.returncode == 0:
        git("add", "-A")
        if git("diff", "--cached", "--name-only"):
            git("commit", "-m", f"{name} run (self-hosted)")
            git("push", "origin", "main")

    status[name] = {"last_run": started, "exit_code": r.returncode,
                    "ok": r.returncode == 0}
    return status[name]


scheduler = BackgroundScheduler(timezone="UTC")
for name, job in JOBS.items():
    scheduler.add_job(run_job, CronTrigger.from_crontab(job["cron"]),
                      args=[name], id=name, misfire_grace_time=3600)
scheduler.start()

app = FastAPI(title="NFL Model Server")


@app.get("/api/status")
def api_status():
    return {name: {**status[name],
                   "schedule": JOBS[name]["cron"],
                   "next_run": str(scheduler.get_job(name).next_run_time)}
            for name in JOBS}


@app.post("/api/run/{name}")
def api_run(name: str):
    if name not in JOBS:
        raise HTTPException(404, f"unknown job; choose from {list(JOBS)}")
    return run_job(name)


@app.get("/api/log/{name}", response_class=PlainTextResponse)
def api_log(name: str):
    f = LOG_DIR / f"{name}.log"
    return f.read_text() if f.exists() else "no runs yet"


@app.get("/")
def index():
    return FileResponse(REPO / "docs" / "index.html")


app.mount("/", StaticFiles(directory=REPO / "docs"), name="docs")
