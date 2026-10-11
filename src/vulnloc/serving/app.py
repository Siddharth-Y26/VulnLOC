"""The FastAPI application (``PRD.md`` §5.6; ``docs/Phases.md`` Phase 9).

Three endpoints the PRD names and one it asks for as a "should":

* ``POST /analyze`` — one function in, the §5.4 object out, and nothing else in the body.
  What the contract has no field for travels in headers: the run that answered, how many
  lines were scored, how many were not (past the line cap, or without features);
* ``POST /analyze/report`` — the same analysis with the reader's report beside the §5.4
  object: a plain summary, the highest-ranked lines whether or not any was flagged, every
  flagged line's verified explanation, the weakness patterns that were tested, and what
  was not examined. ``/analyze`` itself stays exactly the contract;
* ``POST /analyze/batch`` — up to :data:`~vulnloc.serving.schemas.MAX_BATCH_ITEMS`
  functions, analyzed one after another, each with its own result or its own error;
* ``GET /health`` and ``GET /version`` — the second says exactly which checkpoint is
  answering: run id, git sha, config hash, threshold;
* ``GET /`` — a one-page demo (``docs/Design.md`` §5) that calls the endpoints above from
  the browser. It adds no capability the API does not have.

Every failure leaves in one shape (``Rules.md`` §4.7), with the status its class deserves:
input that cannot be parsed or turned into a graph is the client's (400), a request that
does not validate is 422, and anything the service itself got wrong is 500. **No failure
is turned into a clean-looking result**: a function whose graph will not build is an
error, not a semantic-only answer (``Rules.md`` §4.8).

The model is loaded once, at start-up, from the bundle named by ``VULNLOC_BUNDLE`` (or
handed to :func:`create_app`), and a service with no usable bundle does not start.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from importlib import resources
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse

import vulnloc
from vulnloc.serving.inference import Analysis, Analyzer
from vulnloc.serving.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchItem,
    BatchRequest,
    BatchResponse,
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    ReportResponse,
    VersionResponse,
)
from vulnloc.utils.exceptions import (
    ConfigError,
    DataError,
    GraphBuildError,
    ParseError,
    VulnLocError,
)
from vulnloc.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["BUNDLE_ENV", "CORPUS_LANGUAGE", "create_app", "status_for"]

_LOGGER = get_logger(__name__)

BUNDLE_ENV = "VULNLOC_BUNDLE"
"""Environment variable naming the bundle directory to serve."""

CORPUS_LANGUAGE = {"c": "C", "cpp": "CPP"}
"""The request's language, as the corpus spells it (``CodeUnit.language``)."""

_CLIENT_ERRORS: tuple[type[VulnLocError], ...] = (ParseError, GraphBuildError, DataError)

DEMO_PAGE = resources.files("vulnloc.serving") / "static" / "index.html"
"""The one-page demo (``docs/Design.md`` §5): paste a function, see its lines ranked."""

_DESCRIPTION = """
Ranks the lines of one C or C++ function by how likely each is to be part of a
vulnerability, and explains the lines it flags.

**A result with `is_vulnerable: false` means that no line reached the threshold. It is
not evidence that the function is safe.** Every model behind this API was trained on
vulnerable functions only. This is a decision-support tool; it gives no security
guarantee. See the model card shipped with the bundle.
"""

_ERRORS: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "The code could not be parsed or graphed."},
    422: {"model": ErrorResponse, "description": "The request does not validate."},
    500: {"model": ErrorResponse, "description": "The service failed."},
}


def status_for(error: VulnLocError) -> int:
    """The HTTP status an error of ours leaves with.

    Args:
        error: The error.

    Returns:
        400 for input the pipeline cannot read, 500 for everything else.
    """
    return 400 if isinstance(error, _CLIENT_ERRORS) else 500


def _error(status: int, kind: str, message: str, unit_id: str | None = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(type=kind, message=message, unit_id=unit_id))
    return JSONResponse(status_code=status, content=body.model_dump())


def _version(analyzer: Analyzer) -> VersionResponse:
    manifest = analyzer.bundle.manifest
    verification = manifest.get("verification") or {}
    return VersionResponse(
        vulnloc_version=vulnloc.__version__,
        run_id=str(manifest["run_id"]),
        git_sha=str(manifest["git_sha"]),
        config_hash=str(manifest["config_hash"]),
        seed=int(manifest.get("seed", 0)),
        split_fingerprint=str(manifest.get("split_fingerprint", "")),
        fusion=str(manifest["fusion"]["kind"]),
        threshold=float(manifest["calibration"]["threshold"]),
        calibrator=str(manifest["calibration"]["kind"]),
        strip_comments=bool(manifest["text"]["strip_comments"]),
        line_cap=int(manifest["llm"]["n_lines"]),
        exported_at=str(manifest.get("exported_at", "")),
        max_logit_deviation=verification.get("max_logit_deviation"),
    )


def create_app(analyzer: Analyzer | None = None, *, bundle_dir: str | None = None) -> FastAPI:
    """Build the application.

    Args:
        analyzer: A loaded analyzer. When ``None`` one is loaded at start-up.
        bundle_dir: The bundle to load when no analyzer is given; defaults to the
            ``VULNLOC_BUNDLE`` environment variable.

    Returns:
        The application. Start-up raises :class:`~vulnloc.utils.exceptions.ConfigError`
        if there is neither an analyzer nor a bundle to load one from.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        loaded = analyzer
        if loaded is None:
            root = bundle_dir or os.environ.get(BUNDLE_ENV)
            if not root:
                raise ConfigError(
                    "no serving bundle configured",
                    hint=f"Set {BUNDLE_ENV} to a directory built by "
                    "scripts/export_serving_bundle.py.",
                )
            loaded = Analyzer.load(root)
            _LOGGER.info("Bundle loaded", extra={"bundle": root, "run_id": loaded.bundle.run_id})
        app.state.analyzer = loaded
        yield

    app = FastAPI(
        title="VulnLoc",
        description=_DESCRIPTION,
        version=vulnloc.__version__,
        lifespan=lifespan,
    )

    @app.exception_handler(VulnLocError)
    async def _ours(_: Request, error: VulnLocError) -> JSONResponse:
        status = status_for(error)
        if status >= 500:
            _LOGGER.error("Request failed", extra={"error": str(error), "unit_id": error.unit_id})
        return _error(status, type(error).__name__, error.message, error.unit_id)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, error: RequestValidationError) -> JSONResponse:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'][1:]) or 'body'}: {item['msg']}"
            for item in error.errors()
        )
        return _error(422, "ValidationError", problems)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, error: Exception) -> JSONResponse:
        # The last boundary: logged with its traceback, never swallowed, and still answered
        # in the one error shape instead of the framework's plain-text 500.
        _LOGGER.exception("Unhandled error while serving a request", exc_info=error)
        return _error(500, "InternalError", "The service failed to answer this request.")

    def run(request: AnalyzeRequest, app_analyzer: Analyzer) -> Analysis:
        return app_analyzer.analyze(
            request.code, language=CORPUS_LANGUAGE[request.language], unit_id=request.unit_id
        )

    def stamp(response: Response, app_analyzer: Analyzer, diagnostics: dict[str, Any]) -> None:
        response.headers["X-VulnLoc-Run"] = app_analyzer.bundle.run_id[:8]
        response.headers["X-VulnLoc-Lines-Scored"] = str(diagnostics["lines_scored"])
        response.headers["X-VulnLoc-Lines-Beyond-Cap"] = str(diagnostics["lines_beyond_cap"])
        response.headers["X-VulnLoc-Lines-Without-Features"] = str(
            diagnostics["lines_without_features"]
        )
        response.headers["X-VulnLoc-Seconds"] = f"{diagnostics['seconds']:.3f}"

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def demo() -> str:
        """The demo page. Static: it talks to this API from the browser."""
        return DEMO_PAGE.read_text(encoding="utf-8")

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health(request: Request) -> HealthResponse:
        """Whether the service is up and a model is loaded."""
        return HealthResponse(model_loaded=getattr(request.app.state, "analyzer", None) is not None)

    @app.get("/version", response_model=VersionResponse, tags=["service"])
    def version(request: Request) -> VersionResponse:
        """Which checkpoint is answering."""
        return _version(request.app.state.analyzer)

    @app.post("/analyze", response_model=AnalyzeResponse, responses=_ERRORS, tags=["analysis"])
    def analyze(body: AnalyzeRequest, request: Request, response: Response) -> dict[str, Any]:
        """Rank and explain the lines of one function."""
        app_analyzer: Analyzer = request.app.state.analyzer
        analysis = run(body, app_analyzer)
        stamp(response, app_analyzer, analysis.diagnostics)
        return analysis.contract

    @app.post(
        "/analyze/report", response_model=ReportResponse, responses=_ERRORS, tags=["analysis"]
    )
    def analyze_report(
        body: AnalyzeRequest, request: Request, response: Response
    ) -> dict[str, Any]:
        """The same analysis, with what a reader needs beside the result.

        Nothing is flagged here that ``/analyze`` does not flag, and no line is explained
        that the verifier did not pass.
        """
        app_analyzer: Analyzer = request.app.state.analyzer
        analysis = run(body, app_analyzer)
        stamp(response, app_analyzer, analysis.diagnostics)
        return {"result": analysis.contract, "report": analysis.report}

    @app.post("/analyze/batch", response_model=BatchResponse, responses=_ERRORS, tags=["analysis"])
    def analyze_batch(body: BatchRequest, request: Request) -> BatchResponse:
        """Rank and explain several functions; each gets a result or its own error."""
        app_analyzer: Analyzer = request.app.state.analyzer
        results = []
        for item in body.items:
            try:
                contract = run(item, app_analyzer).contract
            except VulnLocError as error:
                # One function's failure is that function's answer, not the batch's.
                _LOGGER.warning(
                    "A batch item failed", extra={"unit_id": item.unit_id, "error": str(error)}
                )
                results.append(
                    BatchItem(
                        error=ErrorBody(
                            type=type(error).__name__, message=error.message, unit_id=item.unit_id
                        )
                    )
                )
                continue
            results.append(BatchItem(result=AnalyzeResponse.model_validate(contract)))
        return BatchResponse(results=results)

    return app
