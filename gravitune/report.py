"""Turn a sweep into a tuned config artifact and a human-readable report."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .score import Objective, baseline, rank
from .sweep import Measurement, Target


def tuned_config(target: Target, best: Measurement, base: Measurement | None,
                 objective: Objective) -> dict:
    """The reusable artifact: what to actually run, and the evidence for it."""
    cfg = best.config
    doc = {
        "schema": "gravitune/tuned-config/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": {
            "cpu_part": target.cpu_part,
            "midr": target.midr,
            "cores": target.cores,
            "features": target.features,
            "instance_type": target.instance_type,
            "kernel": target.kernel,
        },
        "objective": {"name": objective.name, "description": objective.description},
        "recommended": {
            "threads": cfg.threads,
            "batch": cfg.batch,
            "ubatch": cfg.ubatch,
            "flash_attn": cfg.flash_attn,
            "model": Path(cfg.model).name,
        },
        "measured": {
            "prefill_tps": round(best.prefill_tps, 2),
            "decode_tps": round(best.decode_tps, 2),
            "ttft_ms": round(best.ttft_ms, 1),
        },
        "llama_cli_args": (
            f"-t {cfg.threads} -b {cfg.batch} -ub {cfg.ubatch} -fa {cfg.flash_attn}"
        ),
    }

    if base is not None:
        doc["baseline"] = {
            "threads": base.config.threads,
            "prefill_tps": round(base.prefill_tps, 2),
            "decode_tps": round(base.decode_tps, 2),
            "ttft_ms": round(base.ttft_ms, 1),
        }
        doc["improvement"] = {
            "prefill_x": round(best.prefill_tps / base.prefill_tps, 3) if base.prefill_tps else None,
            "decode_x": round(best.decode_tps / base.decode_tps, 3) if base.decode_tps else None,
            "ttft_x": round(base.ttft_ms / best.ttft_ms, 3) if best.ttft_ms else None,
        }

    return doc


def write_results(path: Path, target: Target, measurements: Sequence[Measurement]) -> None:
    payload = {
        "target": asdict(target),
        "measurements": [m.to_dict() for m in measurements],
    }
    path.write_text(json.dumps(payload, indent=2))


def markdown_report(target: Target, measurements: Sequence[Measurement],
                    objective: Objective) -> str:
    ranked = rank(measurements, objective)
    base = baseline(measurements, target.cores)
    lines: list[str] = []

    a = lines.append
    a(f"# GraviTune report — {target.cpu_part}")
    a("")
    a(f"- **Target**: {target.cpu_part} (MIDR {target.midr}), {target.cores} cores"
      + (f", `{target.instance_type}`" if target.instance_type else ""))
    a(f"- **ISA features**: `{' '.join(target.features)}`")
    a(f"- **Objective**: `{objective.name}` — {objective.description}")
    a(f"- **Kernel**: {target.kernel}")
    a("")

    if not ranked:
        a("> No successful measurements. Nothing to recommend.")
        return "\n".join(lines)

    best_m = ranked[0][0]
    a("## Recommendation")
    a("")
    a(f"```\n{best_m.config.threads} threads, batch {best_m.config.batch}, "
      f"ubatch {best_m.config.ubatch}, flash-attn {best_m.config.flash_attn}\n```")
    a("")
    if base is not None and base.decode_tps:
        a(f"**{best_m.decode_tps / base.decode_tps:.2f}x decode throughput** vs the "
          f"stock all-core default ({base.decode_tps:.1f} -> {best_m.decode_tps:.1f} tok/s).")
        a("")

    a("## Full sweep")
    a("")
    a("| rank | config | prefill tok/s | decode tok/s | TTFT ms | score |")
    a("|---:|---|---:|---:|---:|---:|")
    for i, (m, s) in enumerate(ranked, 1):
        a(f"| {i} | `{m.config.label()}` | {m.prefill_tps:.1f} | "
          f"{m.decode_tps:.1f} | {m.ttft_ms:.0f} | {s:.1f} |")
    a("")

    failed = [m for m in measurements if not m.ok]
    if failed:
        a("## Configs that failed to run")
        a("")
        a("Listed explicitly — a config that errored is absent data, not a slow result.")
        a("")
        for m in failed:
            a(f"- `{m.config.label()}`: {m.error}")
        a("")

    return "\n".join(lines)
