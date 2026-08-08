"""Focused tests for the subscription-backed ``codex-cli`` backend."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from graphify import llm


def _graph_json() -> str:
    return json.dumps({
        "nodes": [
            {"id": "guide", "label": "Guide", "file_type": "document", "source_file": "guide.md"},
            {"id": "setup", "label": "Setup", "file_type": "concept", "source_file": "guide.md"},
        ],
        "edges": [
            {"source": "guide", "target": "setup", "relation": "explains",
             "confidence": "EXTRACTED", "confidence_score": 1.0},
        ],
        "hyperedges": [],
    })


def test_run_codex_cli_uses_isolated_spark_xhigh_and_schema(monkeypatch):
    monkeypatch.setenv("GRAPHIFY_CODEX_CLI_EFFORT", "xhigh")

    def fake_run(argv, **kwargs):
        output = Path(argv[argv.index("--output-last-message") + 1])
        output.write_text(_graph_json(), encoding="utf-8")
        return MagicMock(
            returncode=0,
            stdout=json.dumps({
                "type": "turn.completed",
                "usage": {"input_tokens": 21, "output_tokens": 8},
            }),
            stderr="",
        )

    with patch("shutil.which", return_value="/fake/codex"), \
         patch("subprocess.run", side_effect=fake_run) as run:
        text, usage = llm._run_codex_cli(
            "extract",
            model="gpt-5.3-codex-spark",
            schema=llm._CODEX_EXTRACTION_JSON_SCHEMA,
        )

    argv = run.call_args.args[0]
    assert json.loads(text)["nodes"]
    assert usage == {"input_tokens": 21, "output_tokens": 8}
    assert argv[argv.index("--model") + 1] == "gpt-5.3-codex-spark"
    assert 'model_reasoning_effort="xhigh"' in argv
    assert {"--ephemeral", "--ignore-user-config", "--ignore-rules", "read-only", "--output-schema"} <= set(argv)


def test_call_codex_cli_parses_graph_and_usage(monkeypatch):
    monkeypatch.setattr(llm, "_run_codex_cli", lambda *a, **kw: (
        _graph_json(), {"input_tokens": 21, "output_tokens": 8}
    ))
    result = llm._call_codex_cli("source")
    assert len(result["nodes"]) == 2
    assert len(result["edges"]) == 1
    assert result["input_tokens"] == 21
    assert result["output_tokens"] == 8
    assert result["model"] == "gpt-5.3-codex-spark"


def test_extract_files_direct_dispatches_without_api_key(tmp_path, monkeypatch):
    source = tmp_path / "guide.md"
    source.write_text("# Guide\n\nSetup explains installation.\n", encoding="utf-8")
    call = MagicMock(return_value={
        **json.loads(_graph_json()),
        "input_tokens": 1,
        "output_tokens": 1,
        "finish_reason": "stop",
    })
    monkeypatch.setattr(llm, "_call_codex_cli", call)
    result = llm.extract_files_direct([source], backend="codex-cli", root=tmp_path)
    assert call.called
    assert result["nodes"]


def test_image_is_attached_by_path(tmp_path):
    image = tmp_path / "diagram.png"
    image.write_bytes(b"not-a-real-image")

    def fake_run(argv, **kwargs):
        Path(argv[argv.index("--output-last-message") + 1]).write_text(_graph_json(), encoding="utf-8")
        return MagicMock(returncode=0, stdout="", stderr="")

    ref = llm._ImageRef(path=image, rel="diagram.png", media_type="image/png", raw=None)
    with patch("shutil.which", return_value="/fake/codex"), \
         patch("subprocess.run", side_effect=fake_run) as run:
        llm._run_codex_cli("extract", model="gpt-5.3-codex-spark", images=[ref])
    argv = run.call_args.args[0]
    assert argv[argv.index("--image") + 1] == str(image)


def test_plain_completion_reuses_codex_runner(monkeypatch):
    monkeypatch.setattr(llm, "_run_codex_cli", lambda *a, **kw: (
        "Documentation Pipeline", {"input_tokens": 5, "output_tokens": 2}
    ))
    usage = {}
    assert llm._call_llm("name it", backend="codex-cli", usage_out=usage) == "Documentation Pipeline"
    assert usage == {"input": 5, "output": 2}


def test_backend_is_zero_cost_and_rejects_invalid_effort(monkeypatch):
    assert llm.estimate_cost("codex-cli", 1_000_000, 1_000_000) == 0.0
    monkeypatch.setenv("GRAPHIFY_CODEX_CLI_EFFORT", "impossible")
    with patch("shutil.which", return_value="/fake/codex"), pytest.raises(ValueError, match="must be one of"):
        llm._run_codex_cli("x", model="gpt-5.3-codex-spark")
