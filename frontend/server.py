"""HTTP around :mod:`frontend.pipeline` — the VulnLoc pipeline inspector.

A development instrument, not a product. It exists so the pipeline can be *watched*
working on a real function instead of inferred from test output, and so the gap between
what is built and what is planned is visible at a glance.

**This is not the Phase 9 serving API.** Phase 9 is unstarted and ``Phases.md`` forbids
building ahead, so nothing here lives under ``src/vulnloc`` and nothing here is designed
to be inherited. When Phase 9 lands it should be written against its own requirements;
this tool can then be deleted or kept as a debugging view.

Run it::

    python frontend/server.py
    python frontend/server.py --port 8100

Then open http://127.0.0.1:8000.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

# `python frontend/server.py` puts `frontend/` on sys.path, not the repo root, so the
# `frontend.pipeline` import below would fail. Prepending the root keeps the documented
# command working without requiring `-m` or an editable install of this directory.
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from frontend.pipeline import AnalysisOptions, analyse, sample_functions  # noqa: E402

_ARTIFACTS = _REPO / "artifacts"

app = FastAPI(title="VulnLoc pipeline inspector", docs_url="/api/docs")


class AnalyseRequest(BaseModel):
    """One inspection request."""

    code: str
    policy: str = "best_of_two"
    declared: str = "C"
    graph: str = "cpg"
    edge_types: list[str] = Field(default_factory=list)
    normalize: bool = True
    windowing: bool = True
    max_length: int = Field(default=512, ge=64, le=512)
    stride: int = Field(default=128, ge=0, le=448)


@app.post("/api/analyse")
def api_analyse(request: AnalyseRequest) -> JSONResponse:
    """Run the built stages over one pasted function.

    Args:
        request: The code and the options to run it under.

    Returns:
        The per-stage report, or a 400/500 carrying the real error. Failures are
        surfaced verbatim rather than smoothed into an empty result — a parse that blew
        up is the single most interesting thing this tool can tell you, and
        ``Rules.md`` forbids swallowing it.
    """
    options = AnalysisOptions(
        policy=request.policy,
        declared=request.declared,
        graph=request.graph,
        edge_types=tuple(request.edge_types),
        normalize=request.normalize,
        windowing=request.windowing,
        max_length=request.max_length,
        stride=request.stride,
    )
    try:
        return JSONResponse(analyse(request.code, options))
    except ValueError as exc:
        return JSONResponse({"error": str(exc), "kind": "input"}, status_code=400)
    except Exception as exc:  # noqa: BLE001 - returned to the caller, never hidden
        return JSONResponse(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "kind": "pipeline",
                "traceback": traceback.format_exc(limit=12),
            },
            status_code=500,
        )


@app.get("/api/samples")
def api_samples() -> list[dict[str, str]]:
    """Return the built-in sample functions.

    Returns:
        Named samples with their source.
    """
    return sample_functions()


@app.get("/api/phases")
def api_phases() -> Any:
    """Return the phase manifest driving the pipeline rail.

    Returns:
        The parsed ``phases.json``.
    """
    return json.loads((_HERE / "phases.json").read_text(encoding="utf-8"))


def _read_artifact(*parts: str) -> Any | None:
    """Read one artefact JSON, or ``None`` when it has not been generated.

    A missing artefact is an honest state — the script that writes it has not been run —
    so it is reported as absent rather than defaulted to zeros.

    Args:
        *parts: Path segments below ``artifacts/``.

    Returns:
        The parsed JSON, or ``None``.
    """
    path = _ARTIFACTS.joinpath(*parts)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/corpus")
def api_corpus() -> dict[str, Any]:
    """Return the measured corpus artefacts, each with the run that produced it.

    Every figure the UI shows about the corpus comes from here, so every figure is
    traceable to a logged run — ``Rules.md`` §5.8. Nothing is recomputed for display.

    Returns:
        The artefacts, with ``None`` for any that has not been generated.
    """
    split = _read_artifact("data", "splits", "split_report.json")
    leakage = _read_artifact("representation", "label_leakage.json")
    parse_report = _read_artifact("representation", "report_ast.json")
    shards = _read_artifact("data", "cache", "shards.json")
    return {
        "split": split,
        "leakage": leakage,
        "parse": parse_report,
        "shards": shards,
        "missing": [
            name
            for name, value in (
                ("split_report.json", split),
                ("label_leakage.json", leakage),
                ("report_ast.json", parse_report),
                ("shards.json", shards),
            )
            if value is None
        ],
    }


@app.get("/api/model")
def api_model() -> dict[str, Any]:
    """Report the model state — which is, deliberately, that there is no usable model.

    The inspector never shows a vulnerability score. The one end-to-end training run
    (``1843412a…``) was voided by ADR-006 because it learned the corpus's indentation
    artefact rather than the vulnerability, and Phase 3's ≥3-seed exit criterion is
    unstarted. Until a valid run exists there is nothing whose output could honestly be
    displayed, and this endpoint exists to say so in the interface rather than leave a
    blank panel that reads like a loading state.

    Returns:
        The model availability record.
    """
    return {
        "available": False,
        "reason": "void-run",
        "voided_run": "1843412a445b48c6a21059fb9689bf6f",
        "decision": "ADR-006",
        "headline": "No trained model. No scores are shown anywhere in this tool.",
        "detail": (
            "The only end-to-end run reported val line F1 0.9606 and precision 0.9847 — "
            "above the project's own ~30% line-label-noise ceiling, which makes those "
            "numbers impossible. It had learned a one-line whitespace rule: on the "
            "corpus as shipped, 'does this line start with a space?' recovers the flaw "
            "label at precision 1.0000 with zero false positives on all six partitions. "
            "The corpus has since been normalized and rebuilt, and the three seeds have "
            "not been re-run."
        ),
        "unblocks": "python scripts/train_semantic.py train=semantic seed=42",
    }


app.mount("/", StaticFiles(directory=_HERE / "static", html=True), name="static")


def main() -> None:
    """Parse arguments and serve."""
    parser = argparse.ArgumentParser(description="VulnLoc pipeline inspector")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "frontend.server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
