#!/usr/bin/env python3
"""Summarize an already finished `ninfer-serve` run into the four-column speed table.

Nothing is re-run and no server, GPU, artifact, or network is touched. This reads the files an
earlier http://127.0.0.1:8000 session already left behind in `temp/`:

    temp/ninfer-serve.jsonl   the --request-log-jsonl file; exact timings and speculative rounds
    temp/ninfer-serve.out     the pretty stdout capture; values are re-read from the log lines

When --log is omitted the JSONL is preferred because it is the only source that carries the MTP
round count. Run it and it writes the table to a txt file:

    python3 tests/summarize_serve_request_log.py \
        --log temp/ninfer-serve.jsonl --out temp/measure-serve-table.txt

The output table looks like

    | Prompt tokens | Prefill tok/s | Decode tok/s | MTP acceptance |
    | --- | --- | --- | --- |
    | 25,600 | 7,695.2 | 305.3 | 82.6% (3.43 tok/round) |
    | 25,600 (warm prefix) | - | 305.0 | 82.6% (3.43 tok/round) |
    | 25,602 (thinking off) | 7,680.0 | 211.8 | 50.0% (2.50 tok/round) |

Rows keep the chronological request order and skip short prompts (warmup probes) below
--min-prompt-tokens. Interpretation:

    prefill tok/s  = prompt tokens / prefill seconds
    decode tok/s   = (completion tokens - 1) / decode seconds
    MTP acceptance = accepted / drafted                      (whole request)
    tok/round      = 1 + accepted / speculative rounds

"(warm prefix)" marks a request served from the prefix cache, so it never prefilled and its prefill
rate is printed as "-". "(thinking off)" marks a request that disabled reasoning; real runs decode
far fewer tokens there, which also depresses MTP acceptance.

This is a manual summary helper, not a CTest/pytest case.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
JSONL_LOG = REPO_ROOT / "temp/ninfer-serve.jsonl"
PRETTY_LOG = REPO_ROOT / "temp/ninfer-serve.out"
DEFAULT_OUT = REPO_ROOT / "temp/measure-serve-table.txt"
DEFAULT_MIN_PROMPT_TOKENS = 1024
CACHED_MARGIN_TOKENS = 16
COLUMNS = ("Prompt tokens", "Prefill tok/s", "Decode tok/s", "MTP acceptance")

DONE_LINE = re.compile(r"req#(?P<request_id>\d+)\s+done\b")
STARTED_LINE = re.compile(r"req#(?P<request_id>\d+)\s+started\b.*?\bthinking\s+(?P<state>on|off)\b")
PROMPT_LINE = re.compile(r"prompt\s+(?P<tokens>[\d,]+)")
OUTPUT_LINE = re.compile(r"output\s+(?P<tokens>[\d,]+)")
CACHE_LINE = re.compile(r"cache\s+(?P<tokens>[\d,]+)\s*\(")
RATE_LINE = re.compile(
    r"prefill\s+(?P<prefill>[\d.]+)(?P<prefill_scale>k?)\s+tok/s"
    r".*?decode\s+(?P<decode>[\d.]+)(?P<decode_scale>k?)\s+tok/s"
)
MTP_LINE = re.compile(r"mtp accepted\s+(?P<accepted>\d+)/(?P<drafted>\d+)\s*\(")
ENGINE_READY_LINE = re.compile(r"engine ready\s+\|\s+(?P<model>[^|]+?)\s*\|\s+total\b")
KV_CAPACITY_LINE = re.compile(
    r"capacity\s+\|\s+KV\s+(?P<tokens>[\d,]+)\s+tokens,\s+(?P<dtype>[^,]+),"
)


class SummaryError(RuntimeError):
    """Raised when a saved run cannot be located or interpreted."""


@dataclasses.dataclass
class Row:
    """One completed request, reduced to the four published columns."""

    request_id: int
    prompt_tokens: int
    completion_tokens: int
    prefill_tok_s: float | None
    decode_tok_s: float | None
    prefill_seconds: float | None
    decode_seconds: float | None
    cached_tokens: int
    drafted_tokens: int
    accepted_tokens: int
    rounds: int | None
    rounds_derived: bool
    thinking: bool | None
    warm: bool

    def tag(self) -> str:
        if self.thinking is False:
            return " (thinking off)"
        if self.warm:
            return " (warm prefix)"
        return ""

    def acceptance_percent(self) -> float | None:
        if self.drafted_tokens <= 0:
            return None
        return 100.0 * self.accepted_tokens / self.drafted_tokens

    def tokens_per_round(self) -> float | None:
        if not self.rounds or self.rounds <= 0:
            return None
        return 1.0 + self.accepted_tokens / self.rounds


@dataclasses.dataclass
class Summary:
    """A parsed run: where it came from, how it was launched, and its published rows."""

    path: Path
    kind: str
    exact: bool
    launch: str | None
    measurement: str | None
    rows: list[Row]
    completed_requests: int


def _to_int(text: str | None) -> int | None:
    if text is None:
        return None
    return int(text.replace(",", ""))


def _scaled(text: str | None, scale: str | None) -> float | None:
    if text is None:
        return None
    value = float(text)
    return value * 1000.0 if scale == "k" else value


def _search_int(pattern: re.Pattern[str], line: str) -> int | None:
    match = pattern.search(line)
    return _to_int(match.group("tokens")) if match else None


def _is_warm(cached_tokens: int, prompt_tokens: int) -> bool:
    """A prompt counts as warm only when the prefix cache covered essentially all of it."""
    if prompt_tokens <= 0 or cached_tokens <= 0:
        return False
    if cached_tokens < prompt_tokens - CACHED_MARGIN_TOKENS:
        return False
    return cached_tokens * 10 >= prompt_tokens * 9


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def default_log() -> Path:
    for candidate in (JSONL_LOG, PRETTY_LOG):
        if candidate.is_file():
            return candidate
    raise SummaryError(f"no saved run at {JSONL_LOG} or {PRETTY_LOG}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn a saved ninfer-serve run into the four-column speed table.",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="temp/ninfer-serve.jsonl or temp/ninfer-serve.out (default: first that exists)",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"table file to write (default: {_relative(DEFAULT_OUT)})",
    )
    parser.add_argument(
        "--min-prompt-tokens",
        type=int,
        default=DEFAULT_MIN_PROMPT_TOKENS,
        help=f"drop shorter prompts, e.g. warmup probes (default: {DEFAULT_MIN_PROMPT_TOKENS})",
    )
    return parser.parse_args(argv)


def config_sentence(server_start: dict[str, Any]) -> str | None:
    engine = server_start.get("engine") or {}
    artifact = server_start.get("artifact") or {}
    argv = server_start.get("argv") or []

    model = (engine.get("context_cost") or {}).get("model_id")
    weights = artifact.get("weights_id")
    dtype = str(engine.get("kv_cache") or "").split("-")[0].upper()
    prefill_chunk = engine.get("prefill_chunk")
    max_context = engine.get("max_context")
    backend = engine.get("speculative_backend")
    draft_window = engine.get("speculative_draft_window")
    sampling = "greedy sampling" if "--greedy" in argv else "sampling"

    if not (model and weights and dtype and prefill_chunk and max_context):
        return None

    speculative = f"{str(backend).upper()}{draft_window}" if backend and draft_window else None
    parts = [
        f"{model} ({weights}) used {dtype} KV",
        f"a {prefill_chunk:,}-token prefill chunk",
        f"a {max_context:,}-token context",
    ]
    if speculative:
        parts.append(speculative)
    parts.append(sampling)
    return f"{', '.join(parts[:-1])}, and {parts[-1]}:"


def rows_from_jsonl(events: Iterable[dict[str, Any]], min_prompt: int) -> tuple[list[Row], int]:
    rows: list[Row] = []
    completed = 0

    for event in events:
        if event.get("event") != "request_done":
            continue
        completed += 1

        request = event.get("request") or {}
        result = event.get("result") or {}
        timings = event.get("timings_seconds") or {}
        speculative = event.get("speculative") or {}

        prompt_tokens = int(result.get("prompt_tokens") or 0)
        if prompt_tokens < min_prompt:
            continue

        completion_tokens = int(result.get("completion_tokens") or 0)
        cached_tokens = int(result.get("prefix_cache_hit_tokens") or 0)
        prefill_seconds = float(timings.get("prefill") or 0.0)
        decode_seconds = float(timings.get("decode") or 0.0)
        accepted_tokens = int(speculative.get("accepted_tokens") or 0)
        drafted_tokens = int(speculative.get("drafted_tokens") or 0)
        rounds = int(speculative.get("rounds") or 0) or None
        warm = _is_warm(cached_tokens, prompt_tokens)

        rows.append(
            Row(
                request_id=int(request.get("request_id") or 0),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                prefill_tok_s=None if warm or prefill_seconds <= 0.0 else prompt_tokens / prefill_seconds,
                decode_tok_s=(completion_tokens - 1) / decode_seconds if decode_seconds > 0.0 else None,
                prefill_seconds=prefill_seconds,
                decode_seconds=decode_seconds,
                cached_tokens=cached_tokens,
                drafted_tokens=drafted_tokens,
                accepted_tokens=accepted_tokens,
                rounds=rounds,
                rounds_derived=False,
                thinking=request.get("enable_thinking"),
                warm=warm,
            )
        )

    rows.sort(key=lambda row: row.request_id)
    return rows, completed


def rows_from_pretty(text: str, min_prompt: int) -> tuple[list[Row], int, str | None, str | None]:
    thinking_by_id: dict[int, bool] = {}
    body_by_id: dict[int, str] = {}
    model: str | None = None
    capacity: str | None = None
    completed = 0

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        ready = ENGINE_READY_LINE.search(line)
        if ready:
            model = ready.group("model").strip()
            continue
        kv = KV_CAPACITY_LINE.search(line)
        if kv:
            capacity = f"{_to_int(kv.group('tokens')):,} tokens of {kv.group('dtype')} KV"
            continue

        started = STARTED_LINE.search(line)
        if started:
            thinking_by_id[int(started.group("request_id"))] = started.group("state") == "on"
            continue

        done = DONE_LINE.search(line)
        if not done:
            continue
        completed += 1
        body_by_id[int(done.group("request_id"))] = line

    rows: list[Row] = []
    for request_id, line in body_by_id.items():
        prompt_tokens = _search_int(PROMPT_LINE, line)
        completion_tokens = _search_int(OUTPUT_LINE, line)
        if prompt_tokens is None or completion_tokens is None:
            continue
        if prompt_tokens < min_prompt:
            continue

        cached_tokens = _search_int(CACHE_LINE, line) or 0
        rates = RATE_LINE.search(line)
        mtp = MTP_LINE.search(line)
        accepted_tokens = int(mtp.group("accepted")) if mtp else 0
        drafted_tokens = int(mtp.group("drafted")) if mtp else 0
        warm = _is_warm(cached_tokens, prompt_tokens)

        # The pretty line reports rates but not the speculative round count. One round emits one
        # accepted token plus its own follow-up token, so rounds = output - accepted - 1. That is
        # exact unless the final round was cut short by the stop token.
        derived = completion_tokens - accepted_tokens - 1
        rounds = derived if derived > 0 else None

        rows.append(
            Row(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                prefill_tok_s=None if warm else _scaled(
                    rates.group("prefill") if rates else None,
                    rates.group("prefill_scale") if rates else None,
                ),
                decode_tok_s=_scaled(
                    rates.group("decode") if rates else None,
                    rates.group("decode_scale") if rates else None,
                ),
                prefill_seconds=None,
                decode_seconds=None,
                cached_tokens=cached_tokens,
                drafted_tokens=drafted_tokens,
                accepted_tokens=accepted_tokens,
                rounds=rounds,
                rounds_derived=rounds is not None,
                thinking=thinking_by_id.get(request_id),
                warm=warm,
            )
        )

    rows.sort(key=lambda row: row.request_id)
    return rows, completed, model, capacity


def load(path: Path, min_prompt: int) -> Summary:
    if not path.is_file():
        raise SummaryError(f"{path} does not exist")

    text = path.read_text(encoding="utf-8", errors="replace")
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first.startswith("{"):
        rows, completed, model, capacity = rows_from_pretty(text, min_prompt)
        measurement = None
        if model or capacity:
            bits = " and ".join(part for part in (model, capacity) if part)
            measurement = f"{bits} (launch flags are not recorded in the pretty log)"
        return Summary(
            path=path,
            kind="pretty log, derived",
            exact=False,
            launch=None,
            measurement=measurement,
            rows=rows,
            completed_requests=completed,
        )

    events: list[dict[str, Any]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            events.append(json.loads(stripped))
        except json.JSONDecodeError as error:
            if lineno == len(text.splitlines()):
                continue  # a log that was still being appended to can end mid-record
            raise SummaryError(f"{path}:{lineno}: {error}") from error

    rows, completed = rows_from_jsonl(events, min_prompt)
    server_start = next((event for event in events if event.get("event") == "server_start"), None)
    argv = list(server_start.get("argv") or []) if server_start else []
    return Summary(
        path=path,
        kind="request log, exact",
        exact=True,
        launch=" ".join(argv) if argv else None,
        measurement=config_sentence(server_start) if server_start else None,
        rows=rows,
        completed_requests=completed,
    )


def format_tok_s(value: float | None) -> str:
    return "-" if value is None else f"{value:,.1f}"


def format_acceptance(row: Row) -> str:
    percent = row.acceptance_percent()
    if percent is None:
        return "-"
    tokens_per_round = row.tokens_per_round()
    if tokens_per_round is None:
        return f"{percent:.1f}%"
    return f"{percent:.1f}% ({tokens_per_round:.2f} tok/round)"


def markdown_table(rows: list[Row]) -> list[str]:
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"{row.prompt_tokens:,}{row.tag()}",
                    format_tok_s(row.prefill_tok_s),
                    format_tok_s(row.decode_tok_s),
                    format_acceptance(row),
                )
            )
            + " |"
        )
    return lines


def _rate_part(name: str, seconds: float | None, rate: float | None) -> str | None:
    if seconds is not None:
        return f"{name} {seconds:.3f} s"
    if rate is not None:
        return f"{name} {rate:,.1f} tok/s"
    return None


def detail_line(row: Row) -> str:
    parts = [
        _rate_part("prefill", row.prefill_seconds, row.prefill_tok_s),
        _rate_part("decode", row.decode_seconds, row.decode_tok_s),
        f"completion {row.completion_tokens:,}",
        f"cached {row.cached_tokens:,}",
    ]
    if row.rounds is not None:
        parts.append(f"rounds {row.rounds}" + (" (derived)" if row.rounds_derived else ""))
    parts.append(f"drafted {row.drafted_tokens}, accepted {row.accepted_tokens}")
    return f"req#{row.request_id} {row.prompt_tokens:,}{row.tag() or ' (cold)'}: " + ", ".join(
        part for part in parts if part
    )


def field(label: str, text: str) -> str:
    prefix = f"{label + ':':<14}"
    lines = textwrap.wrap(
        text,
        width=96,
        subsequent_indent=" " * len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not lines:
        return prefix.rstrip()
    return "\n".join([prefix + lines[0], *lines[1:]])


def render(summary: Summary, min_prompt: int) -> str:
    blocks = ["ninfer-serve generation speed summary"]
    blocks.append(field("source", f"{_relative(summary.path)} ({summary.kind})"))
    blocks.append(
        field(
            "rows",
            f"{len(summary.rows)} of {summary.completed_requests} completed requests "
            f"(prompt >= {min_prompt:,} tokens)",
        )
    )
    if summary.launch:
        blocks.append(field("launch", summary.launch))
    if summary.measurement:
        blocks.append(field("measurement", summary.measurement))

    blocks.append("")
    blocks.extend(markdown_table(summary.rows))

    blocks.append("")
    blocks.append("details")
    blocks.append("-------")
    blocks.extend(detail_line(row) for row in summary.rows)

    blocks.append("")
    blocks.append("notes")
    blocks.append("-----")
    blocks.append("- prefill tok/s = prompt tokens / prefill seconds; decode tok/s = tokens after the")
    blocks.append("  first / decode seconds; acceptance = accepted / drafted; tok/round = 1 + accepted")
    blocks.append("  / speculative rounds.")
    blocks.append("- A (warm prefix) row was served from the prefix cache, so it never prefilled and its")
    blocks.append("  prefill rate is printed as -.")
    blocks.append("- A (thinking off) row disabled reasoning, which shortens decode and lowers acceptance.")
    if not summary.exact:
        blocks.append("- Rounds are derived as output - accepted - 1 because the pretty log does not")
        blocks.append("  record them, so tok/round here is approximate.")
        blocks.append("- Pass temp/ninfer-serve.jsonl instead for exact timings and round counts.")

    return "\n".join(blocks)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        path = Path(args.log) if args.log else default_log()
        summary = load(path, args.min_prompt_tokens)
    except SummaryError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if not summary.rows:
        print(
            f"error: no completed request in {path} reaches {args.min_prompt_tokens:,} prompt tokens"
            "; lower --min-prompt-tokens",
            file=sys.stderr,
        )
        return 1

    text = render(summary, args.min_prompt_tokens)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")

    print(text)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
