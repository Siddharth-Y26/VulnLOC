"""L3 — Training: loops, optimizers, schedulers, checkpoints, seeding, tracking.

Contains no architecture logic (``docs/Architecture.md`` §3); models are assembled
from config in L2. Phase 0 provides seeding and MLflow tracking only.
"""

from vulnloc.training.device import accelerator_info, log_accelerator, resolve_device
from vulnloc.training.seeding import seed_everything, seed_worker
from vulnloc.training.tracking import start_run

__all__ = [
    "accelerator_info",
    "log_accelerator",
    "resolve_device",
    "seed_everything",
    "seed_worker",
    "start_run",
]
