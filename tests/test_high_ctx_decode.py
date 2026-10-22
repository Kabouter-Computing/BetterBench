"""High-context decode: measure decode against the prefill sweep's own context.

The load-bearing property is context reuse: each decode prompt must be the
prefill prompt *verbatim* with a task prompt appended, so a backend with prefix
caching reuses the KV it just built and one without re-prefills it. The unit
test below pins that prefix relationship directly; the integration tests cover
the flag surface (disable, auto-disable, category selection) end to end.
"""
from __future__ import annotations

import asyncio
import json

from betterbench.client import RunResult
from betterbench.config import Config
from betterbench.corpus import Prompt
from betterbench import runner


# --------------------------------------------------------------------------- #
# Unit: the decode prompt is a strict extension of the prefill prompt
# --------------------------------------------------------------------------- #
def _ok_result() -> RunResult:
    """A minimal successful run with every field the row builders read."""
    return RunResult(ok=True, ttft_ms=5.0, update_gaps_ms=[5.0, 5.0],
                     decode_tps=100.0, prompt_tokens=100, completion_tokens=20,
                     n_chunks=20, tokens_per_update=1.0, chunking="per_token",
                     pp_tps=1000.0, cached_tokens=0)


def _corpus() -> dict[str, list[Prompt]]:
    return {
        "code": [Prompt(id="c1", category="code", input_len_bucket="short",
                        output_len_bucket="short", max_tokens=64,
                        messages=[{"role": "user", "content": "write a fn"}])],
        "prose": [Prompt(id="p1", category="prose", input_len_bucket="short",
                         output_len_bucket="short", max_tokens=64,
                         messages=[{"role": "user", "content": "write a para"}])],
    }


def test_decode_prompt_is_a_strict_extension_of_the_prefill_prompt(monkeypatch):
    """For every prefill pass, the following decode prompt starts with the
    prefill prompt byte-for-byte and asks for more tokens."""
    calls: list[tuple[list[dict], int]] = []

    async def fake_stream_chat(endpoint, model, messages, **kw):
        calls.append((messages, kw.get("max_tokens", 0)))
        return _ok_result()

    monkeypatch.setattr(runner, "stream_chat", fake_stream_chat)

    cfg = Config()
    cfg.prefill_depths = [200]
    cfg.prefill_runs = 2
    cfg.prefill_warmup = 1
    cfg.high_ctx_decode_tokens = 64
    cfg.run_high_ctx_decode = True

    prefill_rows, hctx_rows = asyncio.run(
        runner.prefill_sweep("http://x", "m", cfg, corpus=_corpus(),
                             high_ctx_categories=["code", "prose"]))

    # One prefill + one decode per pass (warmup + measured) = 6 calls.
    assert len(calls) == 2 * (cfg.prefill_warmup + cfg.prefill_runs)
    for i in range(0, len(calls), 2):
        prefill_msgs, prefill_mt = calls[i]
        decode_msgs, decode_mt = calls[i + 1]
        # The prefill prompt is a strict prefix of the decode prompt.
        assert decode_msgs[:len(prefill_msgs)] == prefill_msgs
        # ...and the decode appends a task prompt and asks for more tokens.
        assert len(decode_msgs) > len(prefill_msgs)
        assert decode_mt > prefill_mt
    # The task prompt rotates across the selected categories.
    assert len(hctx_rows) == 1 and not hctx_rows[0]["skipped"]
    assert hctx_rows[0]["categories"] == ["code", "prose"]


def test_high_ctx_decode_off_sends_no_decode_calls(monkeypatch):
    calls: list[tuple[list[dict], int]] = []

    async def fake_stream_chat(endpoint, model, messages, **kw):
        calls.append((messages, kw.get("max_tokens", 0)))
        return _ok_result()

    monkeypatch.setattr(runner, "stream_chat", fake_stream_chat)
    cfg = Config()
    cfg.prefill_depths = [200]
    cfg.prefill_runs = 2
    cfg.prefill_warmup = 0
    cfg.run_high_ctx_decode = False

    prefill_rows, hctx_rows = asyncio.run(
        runner.prefill_sweep("http://x", "m", cfg, corpus=_corpus(),
                             high_ctx_categories=["code"]))
    # Prefill only: one call per pass, no decode.
    assert len(calls) == cfg.prefill_runs
    assert hctx_rows == []


def test_no_corpus_disables_high_ctx_decode(monkeypatch):
    """A prefill-only run with no corpus skips the decode, not the prefill."""
    calls: list[tuple[list[dict], int]] = []

    async def fake_stream_chat(endpoint, model, messages, **kw):
        calls.append((messages, kw.get("max_tokens", 0)))
        return _ok_result()

    monkeypatch.setattr(runner, "stream_chat", fake_stream_chat)
    cfg = Config()
    cfg.prefill_depths = [200]
    cfg.prefill_runs = 2
    cfg.prefill_warmup = 0
    cfg.run_high_ctx_decode = True

    prefill_rows, hctx_rows = asyncio.run(
        runner.prefill_sweep("http://x", "m", cfg, corpus={},
                             high_ctx_categories=["code"]))
    assert len(calls) == cfg.prefill_runs      # prefill ran
    assert hctx_rows == []                     # decode skipped


def test_decode_rejection_keeps_the_prefill_row_complete(monkeypatch):
    """If the server rejects the decode for context length, the prefill passes
    keep running (the prefill row is complete) and only the decode is skipped."""
    calls: list[tuple[list[dict], int]] = []

    def fake_stream_chat(endpoint, model, messages, **kw):
        # The decode asks for more tokens; reject it with a context-length error.
        if kw.get("max_tokens", 0) > 16:
            calls.append((messages, kw.get("max_tokens", 0)))
            return RunResult(ok=False,
                             error="This model's maximum context length is 100 tokens.")
        calls.append((messages, kw.get("max_tokens", 0)))
        return _ok_result()

    async def fake_async(endpoint, model, messages, **kw):
        return fake_stream_chat(endpoint, model, messages, **kw)

    monkeypatch.setattr(runner, "stream_chat", fake_async)
    cfg = Config()
    cfg.prefill_depths = [200]
    cfg.prefill_runs = 3
    cfg.prefill_warmup = 1
    cfg.run_high_ctx_decode = True

    prefill_rows, hctx_rows = asyncio.run(
        runner.prefill_sweep("http://x", "m", cfg, corpus=_corpus(),
                             high_ctx_categories=["code"]))
    # All prefill passes ran (warmup + measured), even though the decode was
    # rejected on the first pass.
    prefill_calls = [c for c in calls if c[1] <= 16]
    assert len(prefill_calls) == cfg.prefill_warmup + cfg.prefill_runs
    # The prefill row is complete (measured passes only).
    assert len(prefill_rows) == 1 and not prefill_rows[0].get("skipped")
    assert len(prefill_rows[0]["ttft_ms"]) == cfg.prefill_runs
    # The decode row is a clean skip.
    assert len(hctx_rows) == 1 and hctx_rows[0]["skipped"]


# --------------------------------------------------------------------------- #
# Integration: the flag surface, end to end against the mock server
# --------------------------------------------------------------------------- #
FAST = {"warmup": 0, "runs_per_category": 1,
        "concurrency_levels": [1], "concurrency_requests": 1,
        "prefill_depths": [200, 400], "prefill_runs": 2, "prefill_warmup": 1,
        "high_ctx_decode_tokens": 16}


def _run(server, tmp_path, *flags, cfg=None):
    from betterbench.cli import main
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({**FAST, **(cfg or {})}))
    out = tmp_path / "r.json"
    main(["run", "--endpoint", server(), "--model", "mock",
          "--config", str(cfg_path), "--out", str(out), "--no-html",
          "--prefill", *flags])
    return json.loads(out.read_text())


def test_high_ctx_decode_runs_by_default_with_prefill(server, tmp_path):
    res = _run(server, tmp_path)
    assert res["prefill"]
    assert res["high_ctx_decode"]
    assert res["config"]["run_high_ctx_decode"] is True
    # Without --quick the full main-decode category set is used.
    assert len(res["high_ctx_decode"][0]["categories"]) > 1
    # Every non-skipped depth carries per-run decode data.
    for d in res["high_ctx_decode"]:
        assert not d["skipped"]
        assert d["decode_tps"] and d["runs"] >= 1


def test_no_high_ctx_decode_flag_disables_it(server, tmp_path):
    res = _run(server, tmp_path, "--no-high-ctx-decode")
    assert res["prefill"]
    assert "high_ctx_decode" not in res
    assert res["config"]["run_high_ctx_decode"] is False


def test_no_prefill_auto_disables_high_ctx_decode(server, tmp_path):
    from betterbench.cli import main
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(FAST))
    out = tmp_path / "r.json"
    main(["run", "--endpoint", server(), "--model", "mock",
          "--config", str(cfg_path), "--out", str(out), "--no-html",
          "--no-prefill", "--decode"])
    res = json.loads(out.read_text())
    assert "prefill" not in res
    assert "high_ctx_decode" not in res
    assert res["config"]["run_high_ctx_decode"] is False


def test_quick_uses_only_the_first_category(server, tmp_path):
    res = _run(server, tmp_path, "--quick")
    cats = res["high_ctx_decode"][0]["categories"]
    assert len(cats) == 1
    # The first category is the first in the (sorted) corpus.
    assert cats == [sorted(cats)[0]]


def test_explicit_high_ctx_categories_win(server, tmp_path):
    res = _run(server, tmp_path, "--high-ctx-categories", "code", "json")
    assert res["high_ctx_decode"][0]["categories"] == ["code", "json"]


def test_prefill_only_with_empty_corpus_skips_decode(server, tmp_path):
    """The prefill sweep builds its own prompts, so an empty corpus is fine —
    the high-context decode skips rather than errors."""
    empty = tmp_path / "empty-corpus"
    empty.mkdir()
    res = _run(server, tmp_path, "--corpus", str(empty))
    assert res["prefill"]
    assert "high_ctx_decode" not in res


def test_report_renders_the_high_ctx_section(server, tmp_path):
    from betterbench.html_report import render_html
    from betterbench.report import render_markdown
    res = _run(server, tmp_path)
    md = render_markdown(res)
    assert "Decode at high context" in md
    # The section is keyed by the prefill depth.
    assert "200" in md and "400" in md
    # The HTML report carries the same table.
    assert "Decode at high context" in render_html(res)
