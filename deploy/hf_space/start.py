"""Boot the one site on a Hugging Face Space.

Downloads the weights and reports the site reads into ``artifacts/`` from the private model
repo named by ``VULNLOC_WEIGHTS_REPO`` (read with the ``HF_TOKEN`` secret), then hands the
process to ``frontend/server.py`` on ``0.0.0.0:7860``. The paths inside that repo are the
paths below ``artifacts/`` on the development machine, so ``frontend/`` runs unchanged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

APP = next(p for p in Path(__file__).resolve().parents if (p / "frontend" / "server.py").exists())
"""The directory holding ``frontend/`` — beside this file in the Space image, the repo root
when run from a checkout (a Codespace)."""


def main() -> None:
    """Fetch the artifacts, then replace this process with the server."""
    repo = os.environ.get("VULNLOC_WEIGHTS_REPO")
    if not repo:
        raise SystemExit(
            "VULNLOC_WEIGHTS_REPO is not set. Add it as a Space variable "
            "(Settings -> Variables and secrets), e.g. <user>/vulnloc-weights."
        )
    snapshot_download(
        repo_id=repo,
        repo_type="model",
        local_dir=APP / "artifacts",
        token=os.environ.get("HF_TOKEN"),
    )
    port = os.environ.get("PORT", "7860")
    server = str(APP / "frontend" / "server.py")
    os.execv(sys.executable, [sys.executable, server, "--host", "0.0.0.0", "--port", port])


if __name__ == "__main__":
    main()
