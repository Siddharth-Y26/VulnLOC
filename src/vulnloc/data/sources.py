"""Declarative catalogue of raw corpora (ADR-001).

Every source records *where* it comes from, *what role it plays*, and *what licence and
citation it carries* — `Rules.md` §7.6 requires licensing to be verified and recorded
before a dataset is used, and `Rules.md` §1.1 forbids inventing dataset facts.

This module is data, not behaviour: it contains no download logic (see
:mod:`vulnloc.data.download`) and no parsing logic (see ``vulnloc.data.loaders``).

Note:
    ``sha256`` is ``None`` for every entry because no upstream publishes a checksum for
    the exact artefacts we consume. Rather than invent one, the downloader records the
    hash observed on first fetch into a manifest and verifies every later fetch against
    it. That gives reproducibility without fabricating provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["SOURCES", "DataSource", "DatasetRole", "SourceKind", "sources_for_role"]


class SourceKind(StrEnum):
    """Transport needed to fetch a source."""

    HTTP = "http"
    GDRIVE = "gdrive"
    HUGGINGFACE = "huggingface"


class DatasetRole(StrEnum):
    """The role a corpus plays in the experimental design (ADR-001).

    Roles never share data. A corpus used for training is never used for generalization
    testing, and vice versa.
    """

    PRIMARY = "primary"
    """Training / validation / test for all headline results."""

    LABEL_FILTER = "label_filter"
    """Used only to filter the primary corpus' labels — never trained on."""

    GENERALIZATION = "generalization"
    """Held-out cross-dataset test. Never tuned on."""

    VALIDATION = "validation"
    """Secondary real-world check (e.g. vulnerable-vs-patched pairs)."""

    CONTROLLED = "controlled"
    """Synthetic, exact-line ground truth. Pipeline validation only, never headline."""

    BASELINE_ONLY = "baseline_only"
    """External prior-art reproduction only. Never trains our branches."""


@dataclass(frozen=True)
class DataSource:
    """A single downloadable artefact.

    Args:
        name: Stable key used on the CLI and in the cache path.
        role: Which experimental role this artefact serves.
        kind: Transport required to fetch it.
        locator: URL, Google Drive file id, or Hugging Face repo id.
        filename: Local filename (or directory name, for Hugging Face snapshots).
        dataset: Human-readable corpus name, for reports and the data card.
        license: SPDX-ish licence string. ``"unknown"`` blocks use until resolved.
        citation: The paper to cite. Required — `Rules.md` §7.6.
        notes: Why this artefact and not another distribution of the same corpus.
        sha256: Expected digest, or None if no upstream checksum exists.
        ignore_patterns: Hugging Face sources only — glob patterns to skip. Some
            dataset repos ship tens of thousands of auxiliary files we do not use;
            fetching them wastes hours and trips the Hub's API rate limit.
        aliases: Names this artefact arrives under when downloaded by hand. Upstream
            names are often generic (``train.csv``) and would collide between corpora
            in a shared directory, so they are normalized to :attr:`filename` on
            adoption. Recording them here also documents the upstream provenance.
    """

    name: str
    role: DatasetRole
    kind: SourceKind
    locator: str
    filename: str
    dataset: str
    license: str
    citation: str
    notes: str = ""
    sha256: str | None = None
    ignore_patterns: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


# --- Big-Vul, primary corpus (ADR-001 §D1.1) ---------------------------------------
# Deliberately the LineVul distribution: it is the ONLY one carrying flaw_line and
# flaw_line_index. The original ZeoVan CSV and the bstee615/bigvul HF mirror have
# func_before/func_after only, and would yield a corpus incomparable with every
# published line-level baseline.
_BIGVUL_CITATION = (
    "Fan et al., 'A C/C++ Code Vulnerability Dataset with Code Changes and CVE "
    "Summaries', MSR 2020, doi:10.1145/3379597.3387501; splits from Fu & "
    "Tantithamthavorn, 'LineVul', MSR 2022, doi:10.1145/3524842.3528452"
)
_BIGVUL_NOTE = (
    "LineVul release only — the sole distribution with flaw_line/flaw_line_index. "
    "Split is LineVul's RANDOM split; ADR-001 §D1.3 requires re-splitting project-wise "
    "and temporally before any headline number."
)

SOURCES: dict[str, DataSource] = {
    "bigvul_train": DataSource(
        name="bigvul_train",
        role=DatasetRole.PRIMARY,
        kind=SourceKind.GDRIVE,
        locator="1ldXyFvHG41VMrm260cK_JEPYqeb6e6Yw",
        filename="bigvul_train.csv",
        aliases=("train.csv",),
        dataset="Big-Vul (LineVul release)",
        license="MIT",
        citation=_BIGVUL_CITATION,
        notes=_BIGVUL_NOTE,
    ),
    "bigvul_val": DataSource(
        name="bigvul_val",
        role=DatasetRole.PRIMARY,
        kind=SourceKind.GDRIVE,
        locator="1yggncqivMcP0tzbh8-8Eu02Edwcs44WZ",
        filename="bigvul_val.csv",
        aliases=("val.csv",),
        dataset="Big-Vul (LineVul release)",
        license="MIT",
        citation=_BIGVUL_CITATION,
        notes=_BIGVUL_NOTE,
    ),
    "bigvul_test": DataSource(
        name="bigvul_test",
        role=DatasetRole.PRIMARY,
        kind=SourceKind.GDRIVE,
        locator="1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V",
        filename="bigvul_test.csv",
        aliases=("test.csv",),
        dataset="Big-Vul (LineVul release)",
        license="MIT",
        citation=_BIGVUL_CITATION,
        notes=_BIGVUL_NOTE,
    ),
    # --- PrimeVul: label filter + paired validation (ADR-001 §D1.2, §D1.5) ----------
    # Carries func_hash, which is the join key for both the label filter and the
    # mandatory train-set overlap removal before the paired evaluation.
    "primevul": DataSource(
        name="primevul",
        role=DatasetRole.LABEL_FILTER,
        kind=SourceKind.HUGGINGFACE,
        locator="colin/PrimeVul",
        filename="primevul",
        dataset="PrimeVul",
        license="MIT",
        citation=(
            "Ding et al., 'Vulnerability Detection with Code Language Models: How Far "
            "Are We?', ICSE 2025, arXiv:2403.18624"
        ),
        notes=(
            "Function-level ONLY — no line labels; never use for a line-level number. "
            "Built from Big-Vul/CrossVul/CVEfixes/DiverseVul, so it is NOT a valid "
            "generalization set for a Big-Vul-trained model. The 'paired' subset is the "
            "vulnerable-vs-patched validation set of ADR-001 §D1.5."
        ),
        # The repo ships a `file_contents/` tree of whole source files — one small file
        # per entry, tens of thousands of them. We join on func_hash and never read
        # them; fetching the tree wastes hours and trips the Hub's API rate limit.
        ignore_patterns=("file_contents/*", "file_contents/**"),
    ),
    # --- DiverseVul: external generalization (ADR-001 §D1.4) -----------------------
    "diversevul": DataSource(
        name="diversevul",
        role=DatasetRole.GENERALIZATION,
        kind=SourceKind.GDRIVE,
        locator="12IWKhmLhq7qn5B_iXgn5YerOQtkH-6RG",
        filename="diversevul_20230702.json",
        dataset="DiverseVul",
        license="see wagner-group/diversevul",
        citation=(
            "Chen et al., 'DiverseVul: A New Vulnerable Source Code Dataset for Deep "
            "Learning Based Vulnerability Detection', RAID 2023, "
            "doi:10.1145/3607199.3607242"
        ),
        notes=(
            "Function-level ONLY — generalization claims from it are function-level "
            "claims. Must be restricted to projects absent from Big-Vul before use."
        ),
    ),
    # --- Juliet: controlled, exact-line ground truth (ADR-001 §D1.6) ---------------
    # The only corpus in this design with EXACT line ground truth: manifest.xml names
    # the file and line of every flaw. Pipeline validation and faithfulness tests only,
    # never a headline number (ResearchPlan.md §3.3).
    # Version 1.3.1 (SARD suite 116), not 1.3 — NIST marks suite 112 deprecated.
    "juliet": DataSource(
        name="juliet",
        role=DatasetRole.CONTROLLED,
        kind=SourceKind.HTTP,
        locator=(
            "https://www.nist.gov/SARD/downloads/test-suites/"
            "2022-08-11-juliet-c-cplusplus-v1-3-1-with-extra-support.zip"
        ),
        filename="juliet-c-cplusplus-v1.3.1.zip",
        aliases=("2022-08-11-juliet-c-cplusplus-v1-3-1-with-extra-support.zip",),
        dataset="NIST SARD Juliet C/C++ 1.3.1",
        license="US Government work (NIST) — not subject to copyright in the US",
        citation="NIST Software Assurance Reference Dataset, Juliet C/C++ 1.3.1, test suite 116",
        notes=(
            "64,099 test cases, 118+ CWEs, ~671 MB. manifest.xml gives exact file+line "
            "per flaw. Synthetic — never use for headline or generalization claims."
        ),
        # Published by NIST on the test-suite page; the only upstream checksum in this
        # catalogue, so it is verified against rather than recorded on first fetch.
        sha256="331b288f17ea95076d76e7bcc1d0e307f9e78241b16e744340213c8e2986919a",
    ),
    "diversevul_metadata": DataSource(
        name="diversevul_metadata",
        role=DatasetRole.GENERALIZATION,
        kind=SourceKind.GDRIVE,
        locator="19cJ7avNtsziaYkrrYuW7FeFdvgrxoNLc",
        filename="diversevul_metadata.json",
        aliases=("diversevul_20230702_metadata.json",),
        dataset="DiverseVul (metadata)",
        license="see wagner-group/diversevul",
        citation="Chen et al., RAID 2023, doi:10.1145/3607199.3607242",
        notes=(
            "Commit and repository URLs for 7,512 commits. Required: without it the "
            "project-disjoint restriction of ADR-001 §D1.4 is guesswork."
        ),
    ),
}


def sources_for_role(role: DatasetRole) -> list[DataSource]:
    """Return every source serving ``role``, ordered by name.

    Args:
        role: The experimental role to filter by.

    Returns:
        Matching sources, sorted for deterministic iteration.
    """
    return sorted((s for s in SOURCES.values() if s.role is role), key=lambda s: s.name)
