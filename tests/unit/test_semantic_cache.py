"""ADR-009's cache — an optimization on the critical path of the load-bearing experiment.

Three properties make it safe, and each is a test here because each fails silently:

**The cached path must equal the live frozen path, for every arm** (§D9.6). This is the
gate on the whole change. It runs end to end — shards, tokenizer, the real
``encode_units``, a manifest on disk, ``FusedLineDataset``, ``collate_fused``, the model —
because the ways this can go wrong are in the joins between those: a row sliced one line
short, a unit padded against the wrong axis, a cache read in a different unit order.

**A stale cache must refuse itself, naming what changed** (§D9.4). Every digest input is
mutated in turn. A cache that loaded anyway would train, converge and describe a corpus
that no longer exists.

**Only a deterministic encoder may be cached** (§D9.5). Dropout left on makes the cache one
random draw, and the live path would disagree with it by a margin that looks like noise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from omegaconf import OmegaConf  # noqa: E402

from tests.unit.test_fused_datasets import make_tokenizer, unit, vocab_file  # noqa: E402
from vulnloc.data.cache import write_shards  # noqa: E402
from vulnloc.models.fused import FusedLineModel  # noqa: E402
from vulnloc.models.fusion import build_fusion  # noqa: E402
from vulnloc.models.heads.line_head import LineHead  # noqa: E402
from vulnloc.models.structural import build_structural_encoder  # noqa: E402
from vulnloc.training.datasets import LineLabelledDataset  # noqa: E402
from vulnloc.training.fused_datasets import FusedLineDataset, collate_fused  # noqa: E402
from vulnloc.training.graph_datasets import load_node_vocab  # noqa: E402
from vulnloc.training.semantic_cache import (  # noqa: E402
    SemanticFeatureCache,
    assert_deterministic,
    cache_inputs,
    encode_units,
    write_cache,
)
from vulnloc.utils.exceptions import ConfigError, DataError, ModelError  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    """The byte-level tokenizer ``test_fused_datasets`` uses, so both suites share fixtures."""
    return make_tokenizer()


ARMS = ["cross_attention", "concat", "sum", "gated", "llm_only", "gnn_only"]
SEM_DIM = 12
STR_DIM = 8
FUSED_DIM = 8

EQUIVALENCE_TOLERANCE = 1e-5
"""§D9.6: a constant, not a parameter. Loosening it is a code-review conversation."""

SHORT = "int f(int n) {\n    int x = n + 1;\n    return x;\n}\n"
LONG = (
    "int g(int a, int b) {\n"
    + "".join(f"    a = a + b * {i};\n" for i in range(24))
    + "    return a;\n}\n"
)


class StubEncoder(torch.nn.Module):
    """``PretrainedCodeEncoder``'s contract at test size, attention-mask aware."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        torch.manual_seed(5)
        self.embedding = torch.nn.Embedding(512, SEM_DIM)
        self.mix = torch.nn.Linear(SEM_DIM, SEM_DIM)
        self.dropout = torch.nn.Dropout(dropout)

    @property
    def hidden_size(self) -> int:
        """Width of the token states."""
        return SEM_DIM

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """``[W, T, D]``, with a window-level term so tokens are not independent."""
        hidden = self.embedding(batch["input_ids"])
        mask = batch["attention_mask"].unsqueeze(-1).float()
        context = (hidden * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
        return torch.as_tensor(self.dropout(torch.tanh(self.mix(hidden + context))))


def config() -> Any:
    """The slice of the composed config ``cache_inputs`` reads."""
    return OmegaConf.create(
        {
            "llm": {
                "model_name": "stub",
                "tokenizer_name": "stub",
                "pretrained": True,
                "max_length": 64,
                "stride": 16,
                "windowing": True,
                "n_lines": 512,
                "frozen": True,
            },
            "representation": {"normalize_whitespace": True},
            "train": {"amp": True},
        }
    )


def inputs(**overrides: str) -> dict[str, str]:
    """Digest inputs for the fixture cache, with any key overridden."""
    values = cache_inputs(
        config(),
        split="train",
        split_fingerprint="0ade7483",
        shard_digest="8c1cccdd",
        encoder_revision="rev1",
    )
    return values | overrides


def shards(tmp_path: Path) -> Path:
    """A `train` shard with functions of different lengths, one needing several windows."""
    units = [
        unit("u0", source=SHORT, lines=(2,)),
        unit("u1", source=LONG, lines=(5, 20)),
        unit("u2", source=SHORT, lines=(3,)),
        unit("u3", source=LONG, lines=(26,)),
    ]
    cache = tmp_path / "cache"
    write_shards(units, {u.unit_id: "train" for u in units}, cache)
    return cache


def build_cache(tmp_path: Path, tokenizer: Any, encoder: torch.nn.Module) -> Path:
    """Encode the fixture shard with the real ``encode_units`` and write it to disk."""
    semantic = LineLabelledDataset(shards(tmp_path), "train", tokenizer)
    features = encode_units(
        encoder.eval(),
        semantic.units,
        n_lines=512,
        device=torch.device("cpu"),
        amp=False,
        max_windows=3,  # Several encoder batches, so row slicing is exercised across them.
        dtype=torch.float32,
    )
    out = tmp_path / "semantic_cache"
    write_cache(out, "train", features, inputs())
    return out


def make_model(kind: str, encoder: torch.nn.Module, vocab_size: int) -> FusedLineModel:
    """A small fused model around one arm, sharing ``encoder``."""
    torch.manual_seed(3)
    kwargs: dict[str, Any] = {"sem_dim": SEM_DIM, "str_dim": STR_DIM, "hidden_dim": FUSED_DIM}
    if kind == "cross_attention":
        kwargs |= {"heads": 2, "n_layers": 1}
    return FusedLineModel(
        encoder,  # type: ignore[arg-type]
        build_structural_encoder(
            "gcn", vocab_size=vocab_size, hidden_dim=STR_DIM, depth=2, edge_handling="ignore"
        ),
        build_fusion(kind, **kwargs),
        LineHead(FUSED_DIM, hidden_dim=0, dropout=0.0),
        n_lines=512,
    )


# --- §D9.6: the gate on the whole change ---------------------------------------------


@pytest.mark.parametrize("kind", ARMS)
def test_the_cached_path_equals_the_live_frozen_path_for_every_arm(tmp_path, tokenizer, kind):
    encoder = StubEncoder()
    cache_dir = build_cache(tmp_path, tokenizer, encoder)
    table, meta = load_node_vocab(vocab_file(tmp_path))
    cache = SemanticFeatureCache.load(cache_dir, "train", inputs())

    live_set = FusedLineDataset(tmp_path / "cache", "train", tokenizer, table)
    cached_set = FusedLineDataset(
        tmp_path / "cache", "train", tokenizer, table, semantic_cache=cache
    )
    # Mixed lengths in one batch, so padding to the widest unit is exercised.
    live_batch = collate_fused(live_set.units)
    cached_batch = collate_fused(cached_set.units)
    assert "sem_line_states" in cached_batch and "sem_line_states" not in live_batch

    model = make_model(kind, encoder, meta["size"]).eval()
    live = model(live_batch)
    cached = model(cached_batch)

    assert torch.equal(live["line_mask"], cached["line_mask"])
    assert torch.equal(live["sem_line_mask"], cached["sem_line_mask"])
    assert torch.allclose(live["line_logits"], cached["line_logits"], atol=EQUIVALENCE_TOLERANCE)
    assert torch.allclose(
        live["function_logits"], cached["function_logits"], atol=EQUIVALENCE_TOLERANCE
    )


def test_the_cache_is_actually_used_rather_than_silently_ignored(tmp_path, tokenizer):
    """A cache nobody reads looks exactly like a cache that works (Memory.md §7 #22)."""
    encoder = StubEncoder()
    cache_dir = build_cache(tmp_path, tokenizer, encoder)
    table, meta = load_node_vocab(vocab_file(tmp_path))
    cached_set = FusedLineDataset(
        tmp_path / "cache",
        "train",
        tokenizer,
        table,
        semantic_cache=SemanticFeatureCache.load(cache_dir, "train", inputs()),
    )
    batch = collate_fused(cached_set.units)
    model = make_model("concat", encoder, meta["size"]).eval()
    before = model(batch)["line_logits"]

    # Corrupting the encoder must not change a cached forward pass...
    with torch.no_grad():
        encoder.embedding.weight.mul_(-3.0)
    assert torch.equal(model(batch)["line_logits"], before)
    # ...and corrupting the cached features must.
    batch["sem_line_states"] = batch["sem_line_states"] * -3.0
    assert not torch.allclose(model(batch)["line_logits"], before)


def test_cached_rows_never_reach_past_the_active_axis(tmp_path, tokenizer):
    encoder = StubEncoder()
    cache_dir = build_cache(tmp_path, tokenizer, encoder)
    table, meta = load_node_vocab(vocab_file(tmp_path))
    cached_set = FusedLineDataset(
        tmp_path / "cache",
        "train",
        tokenizer,
        table,
        semantic_cache=SemanticFeatureCache.load(cache_dir, "train", inputs()),
    )
    batch = collate_fused(cached_set.units)
    batch["sem_line_mask"] = torch.nn.functional.pad(batch["sem_line_mask"], (0, 400), value=True)
    batch["sem_line_states"] = torch.nn.functional.pad(batch["sem_line_states"], (0, 0, 0, 400))

    with pytest.raises(ModelError, match="past the active line axis"):
        make_model("concat", encoder, meta["size"]).eval()(batch)


# --- §D9.4: a stale cache refuses itself ---------------------------------------------


@pytest.mark.parametrize("key", sorted(inputs()))
def test_a_cache_built_under_any_different_input_is_refused_by_name(tmp_path, tokenizer, key):
    cache_dir = build_cache(tmp_path, tokenizer, StubEncoder())

    with pytest.raises(ConfigError, match=key.replace(".", r"\.")):
        SemanticFeatureCache.load(cache_dir, "train", inputs(**{key: "something-else"}))


def test_a_missing_cache_is_an_error_not_an_empty_one(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        SemanticFeatureCache.load(tmp_path / "nowhere", "train", inputs())


def test_a_unit_the_cache_never_saw_is_refused(tmp_path, tokenizer):
    cache_dir = build_cache(tmp_path, tokenizer, StubEncoder())
    cache = SemanticFeatureCache.load(cache_dir, "train", inputs())
    del cache.features["u2"]
    table, _ = load_node_vocab(vocab_file(tmp_path))

    with pytest.raises(DataError, match="not in the semantic feature cache"):
        FusedLineDataset(tmp_path / "cache", "train", tokenizer, table, semantic_cache=cache)


def test_a_batch_mixing_cached_and_live_units_is_refused(tmp_path, tokenizer):
    cache_dir = build_cache(tmp_path, tokenizer, StubEncoder())
    table, _ = load_node_vocab(vocab_file(tmp_path))
    live = FusedLineDataset(tmp_path / "cache", "train", tokenizer, table)
    cached = FusedLineDataset(
        tmp_path / "cache",
        "train",
        tokenizer,
        table,
        semantic_cache=SemanticFeatureCache.load(cache_dir, "train", inputs()),
    )

    with pytest.raises(DataError, match="some units"):
        collate_fused([cached.units[0], live.units[1]])


def _train_fused() -> Any:
    """Import ``scripts/train_fused.py`` by path — ``scripts/`` is not a package."""
    import importlib.util
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts" / "train_fused.py"
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location("vulnloc_scripts_train_fused", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_fine_tuned_run_cannot_read_frozen_features():
    """A head trained on features its own encoder never produces, with nothing raising."""
    cfg = OmegaConf.create(
        {"llm": {"frozen": False}, "train": {"semantic_cache": {"enabled": True}}}
    )

    with pytest.raises(ConfigError, match="requires a frozen encoder"):
        _train_fused().semantic_cache_enabled(cfg)


def test_the_live_path_stays_available_for_the_fine_tuned_confirmation():
    """ADR-009 D9.7 needs it: `enabled=false` with an unfrozen encoder is legitimate."""
    cfg = OmegaConf.create(
        {"llm": {"frozen": False}, "train": {"semantic_cache": {"enabled": False}}}
    )

    assert _train_fused().semantic_cache_enabled(cfg) is False


# --- §D9.5: only a deterministic encoder is cached -----------------------------------


def test_an_encoder_with_live_dropout_cannot_be_cached(tmp_path, tokenizer):
    semantic = LineLabelledDataset(shards(tmp_path), "train", tokenizer)
    encoder = StubEncoder(dropout=0.5).train()

    with pytest.raises(ConfigError, match="not deterministic"):
        assert_deterministic(
            encoder, semantic.units[:2], n_lines=512, device=torch.device("cpu"), amp=False
        )


def test_the_same_encoder_in_eval_mode_passes(tmp_path, tokenizer):
    semantic = LineLabelledDataset(shards(tmp_path), "train", tokenizer)
    encoder = StubEncoder(dropout=0.5).eval()

    assert_deterministic(
        encoder, semantic.units[:2], n_lines=512, device=torch.device("cpu"), amp=False
    )
