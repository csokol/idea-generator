"""FastAPI webapp for browsing research pipeline results."""

import argparse
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from webapp.data import get_run, list_runs

TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="Research Results")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Default data dir, overridden by CLI
DATA_DIR = "data"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    runs = list_runs(DATA_DIR)
    return templates.TemplateResponse("index.html", {"request": request, "runs": runs})


@app.get("/run/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: str):
    detail = get_run(run_id, DATA_DIR)
    if detail is None:
        return HTMLResponse("<h1>Run not found</h1>", status_code=404)
    return templates.TemplateResponse("run.html", {"request": request, "run": detail})


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Research results webapp")
    parser.add_argument("--data-dir", default="data", help="Pipeline data directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global DATA_DIR
    DATA_DIR = args.data_dir

    uvicorn.run(app, host=args.host, port=args.port)
