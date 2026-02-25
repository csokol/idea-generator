"""Data loading for the research results webapp."""

import json
from dataclasses import dataclass
from pathlib import Path

import markdown


@dataclass
class ReportFile:
    filename: str
    label: str
    html: str


@dataclass
class RunSummary:
    run_id: str
    goal: str
    started_at: str
    domains: list[str]
    corpus_count: int
    has_report: bool
    report_count: int = 1


@dataclass
class RunDetail:
    run_id: str
    params: dict
    reports: list[ReportFile]
    corpus_count: int


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a file."""
    if not path.exists():
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _find_reports(run_dir: Path) -> list[Path]:
    """Find all report markdown files in a run directory."""
    return sorted(run_dir.glob("report*.md"))


def _report_label(path: Path) -> str:
    """Derive a human-readable label from a report filename.

    report.md -> "default", report-gemini-3pro.md -> "gemini-3pro"
    """
    stem = path.stem  # e.g. "report-gemini-3pro"
    if stem == "report":
        return "default"
    return stem.removeprefix("report-")


def list_runs(data_dir: str = "data") -> list[RunSummary]:
    """Scan data dir for pipeline runs, return sorted by date descending."""
    base = Path(data_dir)
    if not base.is_dir():
        return []

    runs = []
    for params_file in base.glob("*/params.json"):
        run_dir = params_file.parent
        run_id = run_dir.name
        try:
            params = json.loads(params_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        corpus_count = _count_lines(run_dir / "corpus.jsonl")
        reports = _find_reports(run_dir)

        runs.append(RunSummary(
            run_id=run_id,
            goal=params.get("goal", ""),
            started_at=params.get("started_at", ""),
            domains=params.get("domains", []),
            corpus_count=corpus_count,
            has_report=len(reports) > 0,
            report_count=len(reports),
        ))

    runs.sort(key=lambda r: r.started_at, reverse=True)
    return runs


def get_run(run_id: str, data_dir: str = "data") -> RunDetail | None:
    """Load a single run's details, rendering the report to HTML."""
    run_dir = Path(data_dir) / run_id
    params_file = run_dir / "params.json"
    if not params_file.exists():
        return None

    try:
        params = json.loads(params_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    reports = []
    for rpath in _find_reports(run_dir):
        md_text = rpath.read_text()
        html = markdown.markdown(
            md_text, extensions=["tables", "fenced_code", "toc"]
        )
        reports.append(ReportFile(
            filename=rpath.name,
            label=_report_label(rpath),
            html=html,
        ))

    corpus_count = _count_lines(run_dir / "corpus.jsonl")

    return RunDetail(
        run_id=run_id,
        params=params,
        reports=reports,
        corpus_count=corpus_count,
    )
