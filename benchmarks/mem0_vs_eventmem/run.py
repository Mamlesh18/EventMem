"""mem0 vs EventMem: the runner, with a live dashboard.

    # 1. Start Redis WITH the search module (mem0's vector store needs it)
    docker run -d --name eventmem-redis -p 6379:6379 redis/redis-stack:latest

    # 2. Run it
    python -m benchmarks.mem0_vs_eventmem.run

The browser opens on http://localhost:8club and the dashboard fills in as the
benchmark runs. Both systems hit the same Redis with the same embedding model.

Flags:
    --redis-url URL      default redis://localhost:6379
    --port N             dashboard port (default 8077)
    --k N                retrieval cut-off (default 5)
    --no-browser         do not open a browser
    --headless           no server; print to the terminal and write JSON
    --json PATH          also write the full result document
    --mem0-store S       redis (default) or faiss, if RediSearch is missing
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import pathlib
import queue
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from . import corpus
from .phases import comparison_table, run_all, verdicts
from .systems import (
    EMBED_MODEL,
    EventMemMemorySystem,
    EventMemRedisSystem,
    Mem0System,
    new_namespace,
    preflight,
    reset_namespace,
)

HERE = pathlib.Path(__file__).parent
DASHBOARD = HERE / "dashboard.html"

#: Every event emitted during the run, so a browser that connects late still
#: sees the whole story rather than only what happens after it arrived.
_HISTORY: List[Dict[str, Any]] = []
_SUBSCRIBERS: List["queue.Queue"] = []
_LOCK = threading.Lock()


def publish(kind: str, payload: Dict[str, Any]) -> None:
    """Fan one event out to every connected dashboard."""
    # Payload last would let a field called "kind" overwrite the envelope's
    # own kind, which is exactly what happened: system_start events arrived at
    # the dashboard labelled "pull" and "push" and were silently ignored.
    message = {**payload, "kind": kind, "t": time.time()}
    with _LOCK:
        _HISTORY.append(message)
        targets = list(_SUBSCRIBERS)
    for q in targets:
        # A dashboard that closed its tab mid-run must not take the benchmark
        # down with it.
        with contextlib.suppress(Exception):
            q.put_nowait(message)


class Handler(BaseHTTPRequestHandler):
    """Serves the dashboard and streams events over SSE."""

    def log_message(self, *args):  # keep the console readable
        return

    def do_GET(self):  # noqa: N802  (stdlib naming)
        if self.path.startswith("/stream"):
            return self._stream()
        if self.path.startswith("/results"):
            return self._json({"events": _HISTORY})
        return self._dashboard()

    def _dashboard(self):
        try:
            body = DASHBOARD.read_bytes()
        except FileNotFoundError:
            body = b"<h1>dashboard.html is missing</h1>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        q: queue.Queue = queue.Queue()
        with _LOCK:
            backlog = list(_HISTORY)
            _SUBSCRIBERS.append(q)
        try:
            for message in backlog:
                self._send_event(message)
            while True:
                try:
                    message = q.get(timeout=15)
                    self._send_event(message)
                except queue.Empty:
                    # A comment frame keeps proxies and the browser from
                    # treating a quiet stream as a dead one.
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with _LOCK:
                if q in _SUBSCRIBERS:
                    _SUBSCRIBERS.remove(q)

    def _send_event(self, message):
        self.wfile.write(f"data: {json.dumps(message)}\n\n".encode("utf-8"))
        self.wfile.flush()


async def emit(kind: str, payload: Dict[str, Any]) -> None:
    publish(kind, payload)
    # Yield so the SSE threads get scheduled and the dashboard updates while
    # the benchmark is still running rather than all at once at the end.
    await asyncio.sleep(0)


async def benchmark(args, env: Dict[str, Any]) -> Dict[str, Any]:
    namespace = new_namespace()
    await emit("run_start", {
        "namespace": namespace,
        "redis_url": args.redis_url,
        "embed_model": EMBED_MODEL,
        "mem0_store": args.mem0_store,
        "k": args.k,
        "corpus": corpus.summary(),
        "environment": env,
    })

    cleared = await reset_namespace(args.redis_url, namespace)
    await emit("log", {"message": f"namespace {namespace} prepared "
                                  f"({cleared} stale keys removed)"})

    systems = [
        Mem0System(args.redis_url, namespace, vector_store=args.mem0_store),
        EventMemRedisSystem(args.redis_url, namespace),
    ]
    if args.with_in_memory:
        # The ablation: identical runtime, in-process transport and log. Shows
        # what Redis costs EventMem, separately from what EventMem costs
        # against mem0.
        systems.append(EventMemMemorySystem(args.redis_url, namespace))

    results: Dict[str, Any] = {}
    costs: Dict[str, Any] = {}
    for system in systems:
        try:
            results[system.name] = await run_all(system, emit, args.k)
            costs[system.name] = system.costs
        except Exception as exc:
            await emit("system_error", {
                "system": system.name,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise

    rows = comparison_table(results, costs, args.k)
    readings = verdicts(rows, args.k)
    await emit("run_done", {"rows": rows, "verdicts": readings})

    return {
        "namespace": namespace,
        "environment": env,
        "corpus": corpus.summary(),
        "rows": rows,
        "verdicts": readings,
        "detail": {
            name: {phase: {"metrics": r.metrics, "detail": r.detail}
                   for phase, r in phases.items()}
            for name, phases in results.items()
        },
        "events": _HISTORY,
    }


async def selftest() -> Dict[str, Any]:
    """Drive the dashboard with canned numbers and no Redis.

    Exists so the page can be verified independently of the benchmark: if the
    dashboard is broken you want to find that out without first standing up
    Redis, and if a real run looks wrong you want to know whether the fault is
    in the rendering or in the measurement.

    Every figure here is synthetic, and the page says so, so a screenshot of
    this can never be mistaken for a result.
    """
    import random

    await emit("run_start", {
        "namespace": "selftest", "redis_url": "(none)",
        "embed_model": EMBED_MODEL, "mem0_store": "selftest", "k": 5,
        "corpus": corpus.summary(),
        "environment": {"both_on_redis": True, "selftest": True,
                        "embed_model": EMBED_MODEL},
    })
    await emit("log", {"message": "SELFTEST: synthetic numbers, not a result"})

    canned = {
        "mem0": {
            "ingest": {"facts_written": 32, "median_write_ms": 72.0,
                       "mean_write_ms": 78.9, "p95_write_ms": 80.0,
                       "total_wall_ms": 2527.0, "embed_calls": 64, "pushes": 0},
            "retrieval": {"queries": 12, "k": 5, "precision_at_5": 0.417,
                          "recall_at_5": 0.833, "mrr": 1.0,
                          "median_search_ms": 41.2, "mean_search_ms": 42.1,
                          "p95_search_ms": 60.2},
            "propagation": {"required_deliveries": 44, "satisfied": 25,
                            "coverage": 0.568, "retrievals_spent": 11,
                            "retrievals_per_fact_learned": 0.44},
            "reactivity": {"learned_without_asking": 0,
                           "mean_push_latency_ms": float("nan"),
                           "can_react_to_another_agents_write": False},
        },
        "eventmem": {
            "ingest": {"facts_written": 32, "median_write_ms": 9.4,
                       "mean_write_ms": 10.1, "p95_write_ms": 14.0,
                       "total_wall_ms": 420.0, "embed_calls": 32, "pushes": 44},
            "retrieval": {"queries": 12, "k": 5, "precision_at_5": 0.400,
                          "recall_at_5": 0.819, "mrr": 0.958,
                          "median_search_ms": 12.6, "mean_search_ms": 13.0,
                          "p95_search_ms": 18.4},
            "propagation": {"required_deliveries": 44, "satisfied": 44,
                            "coverage": 1.0, "retrievals_spent": 0,
                            "retrievals_per_fact_learned": 0.0},
            "reactivity": {"learned_without_asking": 44,
                           "mean_push_latency_ms": 2.1,
                           "can_react_to_another_agents_write": True},
        },
    }
    costs = {
        "mem0": {"embed_calls": 87, "vector_searches": 23, "llm_calls": 0,
                 "redis_commands_total": 310, "redis_commands": {}},
        "eventmem": {"embed_calls": 44, "vector_searches": 12, "llm_calls": 0,
                     "redis_commands_total": 196, "redis_commands": {}},
    }

    rng = random.Random(7)
    for name in ("mem0", "eventmem"):
        await emit("system_start", {
            "system": name,
            "mode": "pull" if name == "mem0" else "push",
            "label": f"{name} (selftest)",
        })

        await emit("phase_start", {"phase": "ingest", "system": name, "total": 32})
        for i in range(1, 33):
            await emit("ingest_progress", {
                "system": name, "done": i, "total": 32, "fact_id": f"f{i:02d}",
                "latency_ms": round(
                    canned[name]["ingest"]["median_write_ms"] + rng.uniform(-3, 3), 3),
            })
            await asyncio.sleep(0.012)
        await emit("phase_done", {"phase": "ingest", "system": name,
                                  "metrics": canned[name]["ingest"]})

        await emit("phase_start", {"phase": "retrieval", "system": name, "total": 12})
        for q in corpus.QUERIES:
            await emit("retrieval_progress", {
                "system": name, "done": 1, "total": 12, "query_id": q.id,
                "query": q.text, "relevant": sorted(q.relevant),
                "returned": sorted(q.relevant)[:2], "hits": 2,
                "recall": 0.8, "precision": 0.4, "rr": 1.0,
                "latency_ms": round(
                    canned[name]["retrieval"]["median_search_ms"] + rng.uniform(-4, 4), 3),
            })
            await asyncio.sleep(0.05)
        await emit("phase_done", {"phase": "retrieval", "system": name,
                                  "metrics": canned[name]["retrieval"]})

        await emit("phase_start", {"phase": "propagation", "system": name,
                                   "total": len(corpus.AGENTS)})
        for agent in corpus.AGENTS:
            await emit("propagation_progress", {
                "system": name, "agent": agent,
                "searches": 0 if name == "eventmem" else 2,
            })
            await asyncio.sleep(0.06)
        await emit("phase_done", {"phase": "propagation", "system": name,
                                  "metrics": canned[name]["propagation"], "detail": []})

        await emit("phase_start", {"phase": "reactivity", "system": name, "total": 1})
        await emit("phase_done", {"phase": "reactivity", "system": name,
                                  "metrics": canned[name]["reactivity"]})
        await emit("system_done", {"system": name, "costs": costs[name],
                                   "metrics": canned[name]})

    rows = []
    for name in ("mem0", "eventmem"):
        row = {"system": name}
        for metrics in canned[name].values():
            row.update(metrics)
        row["embed_calls"] = costs[name]["embed_calls"]
        row["vector_searches"] = costs[name]["vector_searches"]
        row["llm_calls"] = costs[name]["llm_calls"]
        row["redis_commands"] = costs[name]["redis_commands_total"]
        rows.append(row)

    readings = verdicts(rows, 5)
    await emit("run_done", {"rows": rows, "verdicts": readings})
    return {"rows": rows, "verdicts": readings, "selftest": True,
            "events": _HISTORY}


def print_table(rows: List[Dict[str, Any]], k: int) -> None:
    columns = [
        ("system", "system"),
        (f"recall@{k}", f"recall_at_{k}"),
        (f"prec@{k}", f"precision_at_{k}"),
        ("mrr", "mrr"),
        ("search ms", "median_search_ms"),
        ("write ms", "median_write_ms"),
        ("coverage", "coverage"),
        ("cov/reach", "coverage_of_reachable"),
        ("retrievals", "retrievals_spent"),
        ("no-ask", "learned_without_asking"),
        ("embeds", "embed_calls"),
        ("llm", "llm_calls"),
        ("redis cmds", "redis_commands"),
    ]
    header = [label for label, _ in columns]
    table = []
    for row in rows:
        line = []
        for _, key in columns:
            v = row.get(key)
            if v is None:
                line.append("-")
            elif isinstance(v, float):
                line.append("n/a" if v != v else f"{v:.3f}")
            else:
                line.append(str(v))
        table.append(line)
    widths = [max(len(header[i]), *(len(r[i]) for r in table)) for i in range(len(header))]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("  ".join("-" * w for w in widths))
    for line in table:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(line)))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark mem0's normal retrieval against EventMem's "
                    "event-driven retrieval, both over the same Redis.")
    parser.add_argument("--redis-url", default="redis://localhost:6379")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--headless", action="store_true",
                        help="no dashboard; terminal output only")
    parser.add_argument("--json", help="write the full result document here")
    parser.add_argument("--with-in-memory", action="store_true",
                        help="also run EventMem on the in-process transport, as "
                             "an ablation showing what Redis costs it")
    parser.add_argument("--selftest", action="store_true",
                        help="render the dashboard with synthetic numbers and no "
                             "Redis, to verify the page itself")
    parser.add_argument("--mem0-store", default="redis", choices=["redis", "faiss"],
                        help="mem0's vector store; faiss is the fallback when "
                             "Redis has no search module")
    args = parser.parse_args(argv)

    problems = corpus.validate()
    if problems:
        print("corpus ground truth is inconsistent:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 2

    if args.selftest:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://localhost:{args.port}/"
        print(f"\n  SELFTEST dashboard: {url}")
        print("  Synthetic numbers. This is NOT a benchmark result.\n")
        if not args.no_browser:
            webbrowser.open(url)
        time.sleep(1.5)
        document = asyncio.run(selftest())
        print()
        print_table(document["rows"], 5)
        print("\n  Dashboard live. Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        return 0

    check = preflight(args.redis_url)
    if not check["reachable"]:
        print(f"\nRedis is not reachable at {args.redis_url}", file=sys.stderr)
        print(f"  {check.get('error', '')}\n", file=sys.stderr)
        print("Start it with the search module, which mem0's vector store needs:",
              file=sys.stderr)
        print("  docker run -d --name eventmem-redis -p 6379:6379 "
              "redis/redis-stack:latest\n", file=sys.stderr)
        return 1

    if args.mem0_store == "redis" and not check["redisearch"]:
        print(f"\nRedis at {args.redis_url} is up but has no search module.")
        print(f"  modules present: {check['modules'] or 'none'}")
        print("\nmem0's Redis vector store requires RediSearch. Either:")
        print("  docker rm -f eventmem-redis && docker run -d --name eventmem-redis "
              "-p 6379:6379 redis/redis-stack:latest")
        print("  ...or re-run with --mem0-store faiss, which keeps EventMem on")
        print("  Redis but puts mem0's vectors in-process. That is a weaker")
        print("  comparison and the report will say so.\n")
        return 1

    env = {
        "redis_reachable": True,
        "redisearch": check["redisearch"],
        "redis_modules": check["modules"],
        "mem0_store": args.mem0_store,
        "embed_model": EMBED_MODEL,
        "python": sys.version.split()[0],
        "both_on_redis": args.mem0_store == "redis",
    }

    server = None
    if not args.headless:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://localhost:{args.port}/"
        print(f"\n  Live dashboard: {url}")
        print("  (the page fills in as the benchmark runs)\n")
        if not args.no_browser:
            webbrowser.open(url)
        # Give the browser a moment to connect so it sees the run from the
        # start rather than replaying from history.
        time.sleep(1.5)

    try:
        document = asyncio.run(benchmark(args, env))
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except Exception as exc:
        print(f"\nbenchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if server is not None:
            print("the dashboard is still up; the error is shown there too")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        return 1

    print()
    print_table(document["rows"], args.k)
    print()
    for v in document["verdicts"]:
        print(f"  [{v['kind']}] {v['title']}")
        print(f"      {v['body']}")
    print()

    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps(document, indent=2, default=str), encoding="utf-8")
        print(f"  wrote {args.json}")

    if server is not None:
        print(f"\n  Dashboard still live at http://localhost:{args.port}/")
        print("  Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
