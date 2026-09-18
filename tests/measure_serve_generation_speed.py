#!/usr/bin/env python3
"""Measure prefill, decode, and MTP acceptance on a resident `ninfer-serve` instance.

The printed table corresponds to a configuration sentence such as:

    Qwen3.5-4B (groupwise-int) used FP8 KV, a 4,096-token prefill chunk,
    a 262,144-token context, MTP3, and greedy sampling:

Local launch (terminal 1) -- no rebuild required, only the public Engine serving route:

    ./build/apps/ninfer-serve /path/to/qwen3_5_4b.ninfer \
        --host 127.0.0.1 \
        --port 8000 \
        --model-id Qwen3.5-4B \
        --max-context 262144 \
        --kv-capacity 262144 \
        --prefill-chunk 4096 \
        --kv-dtype fp8 \
        --spec mtp \
        --lm-head-draft \
        --draft-tokens 3 \
        --greedy \
        --device-state-slots 4 \
        --host-state-slots 4 \
        --host-kv-mib 8192 \
        --request-log-jsonl /tmp/ninfer-serve.jsonl

Measure (terminal 2):

    python3 tests/measure_serve_generation_speed.py \
        --base-url http://127.0.0.1:8000 \
        --request-log /temp/ninfer-serve-qwen3-4b.jsonl

What it does
------------
`examples/cli/messages/long_niah_64k.json` holds a 64,512-token NIAH prompt. Its haystack is
truncated and re-fitted with the server's own tokenizer (`/v1/responses/input_tokens`) until the
rendered chat prompt is about 25,600 tokens, so the run stays in the ~25k row of the published
table. `long_niah_256k.json` is 260,096 tokens and is too long for this row.

Three requests are then issued against a single server:

1. cold request                      -- real prefill, decode, and MTP acceptance
2. byte-identical request again       -- compatible-prefix hit, so prefill is reported as "-"
3. same prompt length, thinking off   -- `enable_thinking: false` plus a distinct marker at the
                                         first system token, so the prefix cache cannot serve it
                                         and the prefill number stays meaningful

Do not launch the server with `--no-thinking`: row 1 must keep the server's thinking default so
that row 3 differs from it. Rows are measured on one resident server with prefix reuse enabled
(the default).

MTP `tok/round` needs `speculative.rounds`, which only the request log carries. Without
`--request-log` the script still reports prefill/decode/acceptance from the HTTP `timings` block
and estimates `tok/round` from `drafted_tokens / --draft-tokens`.

The served model id is taken from `GET /v1/models` and is case-sensitive, so a `curl` client must
repeat it verbatim (`Qwen3.5-4B` is rejected with HTTP 404 when the artifact serves
`qwen3.5-4b`). `--vision` is not needed for these rows; it only reserves extra GPU memory.

This is a manual measurement, not a CTest/pytest case: it needs a running server and a real
artifact, and it asserts nothing. See `tests/README.md` for that convention.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
NIAH_SOURCE = REPO_ROOT / "examples/cli/messages/long_niah_64k.json"
DEFAULT_REQUEST_LOG = Path("/tmp/ninfer-serve.jsonl")

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TARGET_TOKENS = 25_600
DEFAULT_TOLERANCE_TOKENS = 64
DEFAULT_MAX_TOKENS = 128
DEFAULT_DRAFT_TOKENS = 3
DEFAULT_SEED = 20_260_918

COUNT_ENDPOINT = "/v1/responses/input_tokens"
CHAT_ENDPOINT = "/v1/chat/completions"
READY_TIMEOUT_SECONDS = 30.0
LOG_EVENT_TIMEOUT_SECONDS = 15.0

# A row is treated as a compatible-prefix hit when nearly the whole prompt is cached.
CACHED_MARGIN_TOKENS = 16
# Local refinement window around the bisection result, in characters.
FIT_WINDOW_CHARS = 4


class MeasurementError(RuntimeError):
    pass


class ServerClient:
    """Minimal HTTP client for one resident ninfer-serve instance."""

    def __init__(self, base_url: str, timeout: float) -> None:
        parts = urllib.parse.urlsplit(base_url if "://" in base_url else "http://" + base_url)
        if parts.scheme != "http":
            raise MeasurementError(f"only http:// base URLs are supported: {base_url!r}")
        self.host = parts.hostname or "127.0.0.1"
        self.port = parts.port or 8000
        self.timeout = timeout

    @property
    def origin(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _request(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        body: bytes | None = None
        headers = {"Accept": "application/json", "Connection": "close"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
        except (OSError, http.client.HTTPException) as exc:
            raise MeasurementError(f"{method} {path} failed: {exc}") from exc
        finally:
            connection.close()
        if response.status != 200:
            detail = raw.decode("utf-8", errors="replace")
            raise MeasurementError(
                f"{method} {path} -> HTTP {response.status} {response.reason}: {detail}"
            )
        try:
            parsed = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MeasurementError(f"{method} {path} returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise MeasurementError(f"{method} {path} returned a non-object JSON body")
        return parsed

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + min(READY_TIMEOUT_SECONDS, self.timeout)
        while True:
            try:
                if self._request("GET", "/health", None).get("status") == "ok":
                    return
            except MeasurementError:
                pass
            if time.monotonic() >= deadline:
                raise MeasurementError(
                    f"no healthy ninfer-serve at {self.origin}; start it first (see this file's "
                    "docstring) or pass --base-url"
                )
            time.sleep(0.5)

    def served_model_id(self) -> str:
        body = self._request("GET", "/v1/models", None)
        data = body.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]:
                    return item["id"]
        raise MeasurementError("GET /v1/models did not report a served model id")

    def count_tokens(self, model_id: str, messages: list[dict[str, str]]) -> int:
        body = self._request("POST", COUNT_ENDPOINT, {"model": model_id, "input": messages})
        value = body.get("input_tokens")
        if not isinstance(value, int) or isinstance(value, bool):
            raise MeasurementError("input token endpoint did not report an integer count")
        return value

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", CHAT_ENDPOINT, payload)


class RequestLogTail:
    """Reads only the serving JSONL records appended after this object was created."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = path.stat().st_size if path.exists() else 0
        self.buffer = b""

    def _read_new(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        if not chunk:
            return []
        self.buffer += chunk
        lines = self.buffer.split(b"\n")
        self.buffer = lines.pop()
        events: list[dict[str, Any]] = []
        for raw_line in lines:
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MeasurementError(f"invalid serving JSONL record in {self.path}: {exc}") from exc
            if isinstance(event, dict):
                events.append(event)
        return events

    def wait_for_request_done(self, prompt_tokens: int) -> dict[str, Any] | None:
        deadline = time.monotonic() + LOG_EVENT_TIMEOUT_SECONDS
        while True:
            events = self._read_new()
            for event in reversed(events):
                name = event.get("event")
                if name == "request_error":
                    message = event.get("error", {}).get("message", "unknown generation error")
                    raise MeasurementError(f"serving request failed: {message}")
                if name != "request_done":
                    continue
                result = event.get("result", {})
                if result.get("prompt_tokens") == prompt_tokens:
                    return event
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)


def load_niah_source() -> tuple[str, str, str]:
    """Return the fixture's system text, its truncatable haystack, and its trailing question."""
    if not NIAH_SOURCE.is_file():
        raise MeasurementError(f"missing fixture: {NIAH_SOURCE}")
    try:
        messages = json.loads(NIAH_SOURCE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"cannot read {NIAH_SOURCE}: {exc}") from exc
    if not isinstance(messages, list) or len(messages) < 2:
        raise MeasurementError("NIAH fixture is not a system/user message pair")
    system = messages[0].get("content")
    user = messages[-1].get("content")
    if not isinstance(system, str) or not isinstance(user, str):
        raise MeasurementError("NIAH fixture messages are not plain text")
    head, separator, question = user.partition("</document>")
    if not separator:
        raise MeasurementError("NIAH fixture has no </document> marker to truncate against")
    return system, head, separator + question


def compose_messages(
    system_prefix: str, source_system: str, head: str, suffix: str, budget_chars: int
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prefix + source_system},
        {"role": "user", "content": head[:budget_chars] + suffix},
    ]


def fit_haystack_budget(
    client: ServerClient,
    model_id: str,
    system_prefix: str,
    source_system: str,
    head: str,
    suffix: str,
    target: int,
    tolerance: int,
) -> tuple[list[dict[str, str]], int, int]:
    """Bisect the haystack character budget until the chat prompt is about `target` tokens."""
    counts: dict[int, int] = {}

    def messages_for(budget: int) -> list[dict[str, str]]:
        return compose_messages(system_prefix, source_system, head, suffix, budget)

    def count(budget: int) -> int:
        cached = counts.get(budget)
        if cached is None:
            cached = client.count_tokens(model_id, messages_for(budget))
            counts[budget] = cached
        return cached

    if count(len(head)) <= target:
        raise MeasurementError(
            f"the whole fixture renders only {count(len(head))} prompt tokens; cannot reach "
            f"{target}. Use a longer NIAH fixture as the source."
        )

    low, high = 0, len(head)
    while low < high:
        middle = (low + high + 1) // 2
        if count(middle) <= target:
            low = middle
        else:
            high = middle - 1

    window = range(max(0, low - FIT_WINDOW_CHARS), min(len(head), low + FIT_WINDOW_CHARS) + 1)
    best = min(window, key=lambda budget: (abs(count(budget) - target), -count(budget)))
    fitted = count(best)
    if abs(fitted - target) > tolerance:
        print(
            f"warning: fitted prompt is {fitted} tokens, {abs(fitted - target)} away from the "
            f"{target} target; the {len(head)}-character haystack limits how finely it can be tuned",
            file=sys.stderr,
        )
    return messages_for(best), fitted, best


def chat_payload(
    model_id: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    seed: int,
    enable_thinking: bool | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "seed": seed,
        "stream": False,
    }
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking
    return payload


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def build_row(
    label: str,
    response: dict[str, Any],
    log_event: dict[str, Any] | None,
    draft_tokens: int,
) -> dict[str, Any]:
    usage = response.get("usage") or {}
    timings = response.get("timings") or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    decode_tokens = max(completion_tokens - 1, 0)

    if log_event is not None:
        seconds = log_event.get("timings_seconds", {})
        speculative = log_event.get("speculative", {})
        try:
            prefill_seconds = float(seconds["prefill"])
            decode_seconds = float(seconds["decode"])
            drafted_tokens = int(speculative["drafted_tokens"])
            accepted_tokens = int(speculative["accepted_tokens"])
            rounds: int | None = int(speculative["rounds"])
            draft_window = int(speculative.get("draft_window", draft_tokens))
            source = "request log"
        except (KeyError, TypeError, ValueError) as exc:
            raise MeasurementError(f"request_done event is missing required metrics: {exc}") from exc
    else:
        prefill_seconds = float(timings.get("prompt_ms", 0.0)) / 1000.0
        decode_seconds = float(timings.get("predicted_ms", 0.0)) / 1000.0
        drafted_tokens = int(timings.get("draft_n", 0))
        accepted_tokens = int(timings.get("draft_n_accepted", 0))
        rounds = None
        draft_window = draft_tokens
        source = "HTTP timings"

    cached_tokens = int(timings.get("cache_n", 0))
    warm = prompt_tokens - cached_tokens <= CACHED_MARGIN_TOKENS

    acceptance = safe_ratio(float(accepted_tokens), float(drafted_tokens))
    if acceptance is not None:
        effective_rounds = rounds
        if effective_rounds is None and draft_window > 0:
            effective_rounds = max(1, round(drafted_tokens / draft_window))
        tokens_per_round = (
            1.0 + accepted_tokens / effective_rounds
            if effective_rounds is not None and effective_rounds > 0
            else None
        )
    else:
        tokens_per_round = None

    return {
        "label": label,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "warm": warm,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "prefill_tok_s": None if warm else safe_ratio(float(prompt_tokens), prefill_seconds),
        "decode_tok_s": safe_ratio(float(decode_tokens), decode_seconds),
        "acceptance": acceptance,
        "tokens_per_round": tokens_per_round,
        "rounds": rounds,
        "drafted_tokens": drafted_tokens,
        "accepted_tokens": accepted_tokens,
        "draft_window": draft_window,
        "source": source,
    }


def format_tok_s(value: float | None) -> str:
    return "-" if value is None else f"{value:,.1f}"


def format_acceptance(row: dict[str, Any]) -> str:
    acceptance = row["acceptance"]
    if acceptance is None:
        return "-"
    tokens_per_round = row["tokens_per_round"]
    if tokens_per_round is None:
        return f"{100.0 * acceptance:.1f}%"
    return f"{100.0 * acceptance:.1f}% ({tokens_per_round:.2f} tok/round)"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def measure(client: ServerClient, log: RequestLogTail | None, model_id: str, args: argparse.Namespace) -> int:
    source_system, head, suffix = load_niah_source()
    prefix_cold = "Benchmark run A. "
    prefix_thinking_off = "Benchmark run B. "
    if len(prefix_cold) != len(prefix_thinking_off):
        raise MeasurementError("row markers must keep the same text length")

    messages, counted, budget = fit_haystack_budget(
        client,
        model_id,
        prefix_cold,
        source_system,
        head,
        suffix,
        args.target_tokens,
        args.tolerance_tokens,
    )
    print(f"served model:   {model_id} ({client.origin})")
    print(f"prompt source:  {NIAH_SOURCE.relative_to(REPO_ROOT)} (fitted to {counted:,} tokens)")
    print(f"metrics source: {'request log ' + str(args.request_log) if log else 'HTTP timings'}")

    warmup = client.chat(
        chat_payload(
            model_id,
            [{"role": "user", "content": "Reply with one short sentence."}],
            16,
            args.seed,
            None,
        )
    )
    if warmup.get("usage") is None:
        raise MeasurementError("warmup response carried no usage block")

    rows: list[dict[str, Any]] = []
    # Row 3 must miss the prefix cache: it re-renders the same haystack under a marker that
    # diverges inside the first prompt tokens, so its prefill number stays meaningful.
    thinking_off_messages = compose_messages(
        prefix_thinking_off, source_system, head, suffix, budget
    )
    plan = (
        ("", messages, None),
        (" (warm prefix)", messages, None),
        (" (thinking off)", thinking_off_messages, False),
    )
    for tag, row_messages, enable_thinking in plan:
        payload = chat_payload(model_id, row_messages, args.max_tokens, args.seed, enable_thinking)
        response = client.chat(payload)
        prompt_tokens = int((response.get("usage") or {}).get("prompt_tokens", -1))
        event = log.wait_for_request_done(prompt_tokens) if log is not None else None
        if log is not None and event is None:
            print(
                "warning: no matching request_done record in the log; falling back to HTTP timings "
                "for this row",
                file=sys.stderr,
            )
        row = build_row(f"{prompt_tokens:,}{tag}", response, event, args.draft_tokens)
        rows.append(row)

    print()
    print(
        markdown_table(
            ("Prompt tokens", "Prefill tok/s", "Decode tok/s", "MTP acceptance"),
            [
                (
                    row["label"],
                    format_tok_s(row["prefill_tok_s"]),
                    format_tok_s(row["decode_tok_s"]),
                    format_acceptance(row),
                )
                for row in rows
            ],
        )
    )
    print()

    for row in rows:
        rounds = "n/a" if row["rounds"] is None else str(row["rounds"])
        print(
            f"{row['label']}: prefill {row['prefill_seconds']:.3f}s, decode "
            f"{row['decode_seconds']:.3f}s, completion {row['completion_tokens']} tokens, "
            f"cached {row['cached_tokens']}, rounds {rounds}, drafted {row['drafted_tokens']}, "
            f"accepted {row['accepted_tokens']}, draft window {row['draft_window']}, "
            f"source {row['source']}"
        )

    if rows[2]["warm"]:
        print(
            "\nwarning: the thinking-off row was served from the prefix cache. Start the server "
            "without --no-thinking so the request-level `enable_thinking: false` changes the prompt."
        )
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"running ninfer-serve origin (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default="",
        help="served model id; defaults to the id reported by GET /v1/models",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=DEFAULT_TARGET_TOKENS,
        help=f"prompt tokens to fit the fixture to (default: {DEFAULT_TARGET_TOKENS})",
    )
    parser.add_argument(
        "--tolerance-tokens",
        type=int,
        default=DEFAULT_TOLERANCE_TOKENS,
        help=f"accepted distance from --target-tokens (default: {DEFAULT_TOLERANCE_TOKENS})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"output budget per request (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--draft-tokens",
        type=int,
        default=DEFAULT_DRAFT_TOKENS,
        help=(
            "MTP draft window the server was started with; used only to estimate tok/round when "
            f"no request log is available (default: {DEFAULT_DRAFT_TOKENS})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"request seed; ignored while the server runs --greedy (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--request-log",
        default=str(DEFAULT_REQUEST_LOG),
        help=(
            "path passed to the server's --request-log-jsonl; needed for exact MTP tok/round "
            f"(default: {DEFAULT_REQUEST_LOG})"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="per-request HTTP timeout in seconds (default: 3600)",
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.target_tokens <= 0 or args.max_tokens <= 0:
        print("error: --target-tokens and --max-tokens must be positive", file=sys.stderr)
        return 1

    client = ServerClient(args.base_url, args.timeout)
    client.wait_until_ready()
    model_id = args.model or client.served_model_id()

    log_path = Path(args.request_log)
    log = RequestLogTail(log_path)
    if not log_path.exists():
        print(
            f"note: no request log at {log_path}; MTP tok/round is estimated. Start the server with "
            f"--request-log-jsonl {log_path} to read `speculative.rounds` directly.",
            file=sys.stderr,
        )
        log = None

    try:
        return measure(client, log, model_id, args)
    except MeasurementError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
