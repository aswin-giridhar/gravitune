"""MCP server exposing GraviTune to coding agents.

Speaks MCP over stdio (JSON-RPC 2.0). Stdlib only, so it runs on a bare Arm box
with no install step.

The point: an agent working on an Arm deployment can ask the *actual machine*
what it is and what config to use, instead of guessing from a model card or
copying an x86 command line. Wire it into Claude Code / Copilot / Codex with:

    {"mcpServers": {"gravitune": {
        "command": "python3", "args": ["-m", "gravitune.mcp_server"],
        "env": {"GRAVITUNE_BENCH": "/path/to/llama-bench"}}}}
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .report import tuned_config
from .score import DEFAULT_OBJECTIVE, OBJECTIVES, baseline, best
from .sweep import SweepRunner, build_grid, detect_target

PROTOCOL_VERSION = "2025-06-18"

TOOLS = [
    {
        "name": "detect_arm_target",
        "description": (
            "Identify the Arm CPU this machine actually is (Neoverse N1/V1/N2/V2 "
            "etc.) from MIDR_EL1, plus core count and ISA features. Use this "
            "before recommending any inference settings, because the correct "
            "settings differ per Arm generation and cloud instance labels do not "
            "tell you which microarchitecture you got."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "check_kernel_dispatch",
        "description": (
            "Report which optimized inference kernel paths this Arm CPU can "
            "reach -- i8mm (int8 matmul), bf16, SVE/SVE2, SME. Answers 'will "
            "KleidiAI's fast int4/int8 GEMM kernels actually engage here, or "
            "silently fall back to a slower path?'"
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "autotune_inference",
        "description": (
            "Sweep llama.cpp inference configs on this Arm machine and return "
            "the best one with measured before/after throughput. Use when asked "
            "to make local LLM inference faster on an Arm server. Takes several "
            "minutes; prefer quick=true for a threads-only sweep."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string",
                               "description": "Absolute path to a GGUF model."},
                "objective": {"type": "string", "enum": sorted(OBJECTIVES),
                              "description": "What to optimise for.",
                              "default": DEFAULT_OBJECTIVE},
                "quick": {"type": "boolean", "default": True,
                          "description": "Threads-only sweep; much faster."},
            },
            "required": ["model_path"],
        },
    },
]


def _text(s: str) -> dict:
    return {"content": [{"type": "text", "text": s}]}


def _err(s: str) -> dict:
    # isError distinguishes "the tool ran and reports a problem" from a
    # transport failure. Agents need that difference to decide whether to retry.
    return {"content": [{"type": "text", "text": s}], "isError": True}


def _tool_detect() -> dict:
    t = detect_target()
    if t.cpu_part == "unknown":
        return _err("Not an Arm target, or /proc/cpuinfo unreadable. "
                    "GraviTune only applies to Arm (aarch64) machines.")
    return _text(json.dumps({
        "cpu_part": t.cpu_part, "midr": t.midr, "cores": t.cores,
        "instance_type": t.instance_type or None, "kernel": t.kernel,
        "isa_features": t.features,
    }, indent=2))


def _tool_dispatch() -> dict:
    t = detect_target()
    if t.cpu_part == "unknown":
        return _err("Not an Arm target.")
    notes = []
    if t.has_i8mm:
        notes.append("i8mm present: KleidiAI int4/int8 GEMM microkernels can engage.")
    else:
        notes.append("i8mm ABSENT: int8 matmul falls back to a slower dot-product "
                     "path. Expect materially lower prefill throughput than an "
                     "i8mm-capable Arm core (e.g. Neoverse V2).")
    if "sve2" in t.features:
        notes.append("SVE2 present.")
    elif t.has_sve:
        notes.append("SVE present (no SVE2).")
    else:
        notes.append("No SVE: NEON-only vectorisation.")
    if "bf16" in t.features:
        notes.append("bf16 present.")
    if "sme" in t.features:
        notes.append("SME present.")
    return _text(json.dumps({
        "cpu_part": t.cpu_part, "isa_features": t.features,
        "i8mm": t.has_i8mm, "sve": t.has_sve,
        "notes": notes,
    }, indent=2))


def _tool_autotune(args: dict) -> dict:
    bench = os.environ.get("GRAVITUNE_BENCH", "")
    if not bench or not Path(bench).exists():
        return _err("llama-bench not found. Set GRAVITUNE_BENCH to its absolute "
                    "path in the MCP server env block.")

    model = args.get("model_path", "")
    if not model or not Path(model).exists():
        return _err(f"model_path not found: {model!r}")

    target = detect_target()
    if target.cores == 0:
        return _err("Could not determine core count.")

    objective = OBJECTIVES[args.get("objective", DEFAULT_OBJECTIVE)]
    grid = build_grid([model], target.cores, quick=bool(args.get("quick", True)))

    runner = SweepRunner(bench)
    measurements = [runner.run_one(c) for c in grid]

    top = best(measurements, objective)
    if top is None:
        errs = "; ".join(f"{m.config.label()}: {m.error}"
                         for m in measurements if not m.ok) or "unknown"
        return _err(f"No config completed successfully. Failures: {errs}")

    doc = tuned_config(target, top, baseline(measurements, target.cores), objective)
    return _text(json.dumps(doc, indent=2))


def handle(req: dict) -> dict | None:
    method = req.get("method")
    rid = req.get("id")

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "gravitune", "version": "0.1.0"},
        }}

    if method in ("notifications/initialized", "initialized"):
        return None  # notification: no response

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}

    if method == "tools/call":
        params = req.get("params", {}) or {}
        name = params.get("name")
        args = params.get("arguments", {}) or {}
        try:
            if name == "detect_arm_target":
                result = _tool_detect()
            elif name == "check_kernel_dispatch":
                result = _tool_dispatch()
            elif name == "autotune_inference":
                result = _tool_autotune(args)
            else:
                return {"jsonrpc": "2.0", "id": rid,
                        "error": {"code": -32601, "message": f"unknown tool: {name}"}}
        except Exception as e:  # surface the real failure, never a bare empty result
            result = _err(f"{type(e).__name__}: {e}")
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    return {"jsonrpc": "2.0", "id": rid,
            "error": {"code": -32601, "message": f"unknown method: {method}"}}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
