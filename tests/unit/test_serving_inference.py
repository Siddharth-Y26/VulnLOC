"""The serving path reads a function the way training read it, or it is scoring something else.

``vulnloc.serving`` may not import the training layer, so it builds its one-function batch
itself. The first test here is the reason that is allowed: tensor for tensor, the batch
:func:`~vulnloc.serving.inference.featurize` builds equals the one ``collate_fused`` builds
for the same function. The rest runs a whole bundle — tiny, randomly initialised, written
to disk and read back — through the path a request takes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from transformers import RobertaConfig  # noqa: E402

from tests.unit.test_fused_datasets import (  # noqa: E402
    SOURCE,
    dataset,
    make_tokenizer,
    unit,
    vocab_file,
)
from vulnloc.explanation.contract import CONTRACT_KEYS, NOT_FLAGGED_TEXT  # noqa: E402
from vulnloc.representation import GRAPH_REGISTRY, CodeUnit, TreeSitterParser  # noqa: E402
from vulnloc.serving.bundle import (  # noqa: E402
    MANIFEST_NAME,
    SCHEMA_VERSION,
    file_digest,
    load_bundle,
)
from vulnloc.serving.inference import Analyzer, build_architecture, featurize  # noqa: E402
from vulnloc.training.fused_datasets import collate_fused  # noqa: E402
from vulnloc.training.graph_datasets import load_node_vocab  # noqa: E402
from vulnloc.utils.exceptions import ConfigError, ParseError  # noqa: E402

LONGER = (
    "int g(char *dst, const char *src, int n) {\n"
    "    int i;\n"
    "    /* copy */\n"
    "    for (i = 0; i < n; i++) {\n"
    "        dst[i] = src[i];\n"
    "    }\n"
    "    return i;\n"
    "}\n"
)
SUBTOKENS = {"n": 2, "x": 3, "dst": 4, "src": 5}
LINE_CAP = 6


# --- the batch ------------------------------------------------------------------------------


@pytest.mark.parametrize("subtokens", [None, SUBTOKENS])
def test_the_served_batch_is_the_batch_training_collates(tmp_path: Path, subtokens: Any) -> None:
    tokenizer = make_tokenizer()
    records = [unit("a"), unit("b", source=LONGER, lines=(5,))]
    fused = dataset(tmp_path, tokenizer, records, subtoken_vocab=subtokens)
    table, _ = load_node_vocab(vocab_file(tmp_path))
    parser, builder = TreeSitterParser(), GRAPH_REGISTRY.get("ast")()

    for index, record in enumerate(records):
        expected = collate_fused([fused[index]])
        batch, _, _ = featurize(
            CodeUnit.from_labelled_function(record),
            tokenizer,
            parser,
            builder,
            table,
            subtokens,
            language_policy="best_of_two",
            n_lines=fused.n_lines,
        )
        # Everything a branch reads is there, and nothing in it differs.
        assert {key for key in expected if key.startswith(("sem_", "str_"))} == {
            key for key in batch if key.startswith(("sem_", "str_"))
        }
        for key, value in batch.items():
            if torch.is_tensor(value):
                assert value.dtype == expected[key].dtype, key
                assert torch.equal(value, expected[key]), key
            else:
                assert value == expected[key], key


# --- a whole bundle ---------------------------------------------------------------------------


def write_bundle(root: Path, *, threshold: float) -> Path:
    """A tiny, randomly initialised bundle with every file a real one has."""
    root.mkdir(parents=True)
    line_tokenizer = make_tokenizer()
    # The Hugging Face tokenizer the wrapper holds: a bundle ships its files.
    backend = line_tokenizer._tokenizer  # noqa: SLF001
    backend.save_pretrained(root / "tokenizer")
    RobertaConfig(
        vocab_size=len(backend),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=48,
        # RoBERTa numbers positions from the padding index up, so a full window needs
        # `max_length + pad_token_id + 1` rows (514 for the real model: 512 + 1 + 1).
        max_position_embeddings=line_tokenizer.max_length + backend.pad_token_id + 1,
        pad_token_id=backend.pad_token_id,
    ).save_pretrained(root / "encoder")
    (root / "node_vocab.json").write_text(
        vocab_file(root).read_text(encoding="utf-8"), encoding="utf-8"
    )
    vocab_file(root).unlink()
    (root / "MODEL_CARD.md").write_text("# A test model\n", encoding="utf-8")
    table = json.loads((root / "node_vocab.json").read_text(encoding="utf-8"))
    manifest: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "run_id": "f" * 32,
        "git_sha": "abc1234",
        "config_hash": "c0ffee",
        "seed": 42,
        "split_fingerprint": "0" * 64,
        "exported_at": "2026-10-10T00:00:00+00:00",
        "text": {"normalize_whitespace": True, "strip_comments": True},
        "llm": {
            "model_name": "test",
            "max_length": line_tokenizer.max_length,
            "stride": line_tokenizer.stride,
            "windowing": True,
            "n_lines": LINE_CAP,
            "dropout": 0.0,
        },
        "graph": {"builder": "ast", "language": "best_of_two"},
        "gnn": {
            "kind": "gcn",
            "hidden_dim": 16,
            "depth": 2,
            "heads": 2,
            "dropout": 0.0,
            "residual": True,
            "layer_norm": True,
            "edge_handling": "auto",
            "node_vocab_size": int(table["size"]),
            "subtoken_vocab_size": 0,
        },
        "fusion": {
            "kind": "cross_attention",
            "args": {"hidden_dim": 16, "heads": 2, "n_layers": 1, "dropout": 0.0},
            "trim_line_axis": True,
        },
        "head": {"hidden_dim": 16, "dropout": 0.0},
        "calibration": {"kind": "platt", "scale": 1.0, "shift": 0.0, "threshold": threshold},
        "explainer": {
            "name": "template",
            "args": {"max_context_lines": 2},
            "evidence": {
                "top_k_attended": 3,
                "min_attention_weight": 0.0,
                "max_dataflow_hops": 2,
                "max_graph_nodes": 5,
                "max_tokens": 6,
            },
        },
    }
    torch.manual_seed(0)
    torch.save(build_architecture(manifest, root / "encoder").state_dict(), root / "model.pt")
    manifest["files"] = {
        path.relative_to(root).as_posix(): file_digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def flags_everything(tmp_path_factory: pytest.TempPathFactory) -> Analyzer:
    """A bundle whose threshold is zero: every line it can score is flagged."""
    return Analyzer.load(write_bundle(tmp_path_factory.mktemp("bundles") / "low", threshold=0.0))


@pytest.fixture(scope="module")
def flags_nothing(tmp_path_factory: pytest.TempPathFactory) -> Analyzer:
    """The same shape of bundle with a threshold no probability reaches."""
    return Analyzer.load(write_bundle(tmp_path_factory.mktemp("bundles") / "high", threshold=1.0))


def test_a_request_comes_back_as_the_contract_with_a_grounded_explanation(
    flags_everything: Analyzer,
) -> None:
    analysis = flags_everything.analyze(SOURCE, unit_id="auth.c::f")
    contract = analysis.contract
    assert tuple(contract) == CONTRACT_KEYS
    assert contract["unit_id"] == "auth.c::f"
    assert contract["is_vulnerable"] is True
    flagged = contract["vulnerable_lines"]
    assert flagged
    assert [entry["score"] for entry in flagged] == sorted(
        (entry["score"] for entry in flagged), reverse=True
    )
    assert contract["risk_score"] == flagged[0]["score"]
    # The code echoed is the line the model saw: stripped at both ends.
    assert all(entry["code"] == entry["code"].strip() for entry in flagged)
    assert contract["explanation"]
    assert analysis.diagnostics["flagged"] == len(flagged)
    assert analysis.diagnostics["explanations_rejected"] == 0


def test_the_same_request_gets_the_same_answer(flags_everything: Analyzer) -> None:
    first, second = (flags_everything.analyze(LONGER) for _ in range(2))
    assert first.contract == second.contract
    assert first.logits.tolist() == second.logits.tolist()


def test_no_flagged_line_is_not_a_statement_of_safety(flags_nothing: Analyzer) -> None:
    contract = flags_nothing.analyze(SOURCE).contract
    assert contract["is_vulnerable"] is False
    assert contract["vulnerable_lines"] == []
    assert contract["explanation"] == NOT_FLAGGED_TEXT
    assert "not evidence that the function is safe" in contract["explanation"]


def test_lines_past_the_cap_are_counted_not_scored(flags_everything: Analyzer) -> None:
    analysis = flags_everything.analyze(LONGER)
    n_lines = len(LONGER.split("\n"))
    assert analysis.diagnostics["lines"] == n_lines
    assert analysis.diagnostics["lines_beyond_cap"] == n_lines - LINE_CAP
    assert analysis.logits.shape == (LINE_CAP,)
    assert all(entry["line"] <= LINE_CAP for entry in analysis.contract["vulnerable_lines"])


def test_a_comment_is_removed_before_the_model_sees_the_line(flags_everything: Analyzer) -> None:
    analysis = flags_everything.analyze(LONGER)
    # Line 3 is a comment and nothing else: blank once stripped, so it carries no code.
    by_line = {entry["line"]: entry["code"] for entry in analysis.contract["vulnerable_lines"]}
    assert by_line.get(3, "") == ""


@pytest.mark.parametrize("code", ["   \n\t\n", "/* only a comment */\n"])
def test_nothing_to_analyze_is_an_error_not_a_clean_result(
    flags_everything: Analyzer, code: str
) -> None:
    with pytest.raises(ParseError, match="no code to analyze"):
        flags_everything.analyze(code, unit_id="empty")


# --- the bundle on disk ---------------------------------------------------------------------


def test_a_bundle_whose_file_changed_is_refused(tmp_path: Path) -> None:
    root = write_bundle(tmp_path / "bundle", threshold=0.5)
    load_bundle(root)
    vocabulary = root / "node_vocab.json"
    vocabulary.write_text(vocabulary.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ConfigError, match="not the file its manifest describes"):
        load_bundle(root)


def test_a_bundle_missing_a_listed_file_is_refused(tmp_path: Path) -> None:
    root = write_bundle(tmp_path / "bundle", threshold=0.5)
    (root / "MODEL_CARD.md").unlink()
    with pytest.raises(ConfigError, match="missing"):
        load_bundle(root)


def test_a_manifest_of_another_schema_or_shape_is_refused(tmp_path: Path) -> None:
    root = write_bundle(tmp_path / "bundle", threshold=0.5)
    path = root / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**manifest, "schema": 99}), encoding="utf-8")
    with pytest.raises(ConfigError, match="schema"):
        load_bundle(root)
    del manifest["calibration"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ConfigError, match="incomplete"):
        load_bundle(root)
    with pytest.raises(ConfigError, match="no serving bundle"):
        load_bundle(tmp_path / "nowhere")


def test_a_file_the_manifest_does_not_list_is_not_read(tmp_path: Path) -> None:
    bundle = load_bundle(write_bundle(tmp_path / "bundle", threshold=0.5))
    (bundle.root / "subtoken_vocab.json").write_text('{"vocab": {"x": 2}}', encoding="utf-8")
    with pytest.raises(ConfigError, match="does not list"):
        bundle.vocabulary("subtoken_vocab.json")


def test_weights_that_do_not_fit_the_manifest_fail_to_load(tmp_path: Path) -> None:
    root = write_bundle(tmp_path / "bundle", threshold=0.5)
    path = root / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["gnn"]["hidden_dim"] = 24
    manifest["fusion"]["args"]["hidden_dim"] = 24
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="size mismatch"):
        Analyzer.load(root)
