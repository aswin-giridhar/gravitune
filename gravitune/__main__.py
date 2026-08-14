"""GraviTune CLI.

    python3 -m gravitune tune  --bench ./llama-bench --models a.gguf [b.gguf ...]
    python3 -m gravitune detect
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .report import markdown_report, tuned_config, write_results
from .score import DEFAULT_OBJECTIVE, OBJECTIVES, baseline, best
from .sweep import SweepRunner, build_grid, detect_target

import json


def cmd_detect(args: argparse.Namespace) -> int:
    t = detect_target()
    print(f"CPU part      : {t.cpu_part} (MIDR {t.midr})")
    print(f"Cores         : {t.cores}")
    print(f"Instance type : {t.instance_type or '(not a cloud instance)'}")
    print(f"Kernel        : {t.kernel}")
    print(f"ISA features  : {' '.join(t.features) or '(none detected)'}")
    print()
    if t.cpu_part == "unknown":
        print("!! Not an Arm target, or /proc/cpuinfo is unreadable.")
        return 1
    print("Kernel dispatch implications:")
    print(f"  i8mm (int8 matmul, KleidiAI int4/int8 GEMM) : "
          f"{'YES' if t.has_i8mm else 'NO  <-- falls back to slower path'}")
    print(f"  SVE                                        : "
          f"{'YES' if t.has_sve else 'NO'}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    bench = Path(args.bench)
    if not bench.exists():
        print(f"error: llama-bench not found at {bench}", file=sys.stderr)
        return 2

    models = [str(Path(m).resolve()) for m in args.models]
    for m in models:
        if not Path(m).exists():
            print(f"error: model not found: {m}", file=sys.stderr)
            return 2

    target = detect_target()
    if target.cores == 0:
        print("error: could not determine core count", file=sys.stderr)
        return 2

    objective = OBJECTIVES[args.objective]
    grid = build_grid(models, target.cores, quick=args.quick)

    # flush=True throughout: a sweep is long-running and usually redirected to a
    # log file, where Python's block buffering would otherwise show nothing at
    # all until the first config finishes. An empty log reads as "it crashed".
    print(f"Target : {target.cpu_part}, {target.cores} cores "
          f"({target.instance_type or 'local'})", flush=True)
    print(f"ISA    : {' '.join(target.features)}", flush=True)
    print(f"Grid   : {len(grid)} configs | objective: {objective.name}", flush=True)
    print(flush=True)

    runner = SweepRunner(str(bench), prompt_tokens=args.prompt_tokens,
                         gen_tokens=args.gen_tokens, reps=args.reps)
    measurements = list(runner.sweep(grid))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    write_results(outdir / "sweep.json", target, measurements)

    top = best(measurements, objective)
    if top is None:
        print("\nNo config completed successfully — nothing to recommend.",
              file=sys.stderr)
        return 1

    base = baseline(measurements, target.cores)
    cfg_doc = tuned_config(target, top, base, objective)

    tag = target.cpu_part.lower()
    if target.instance_type:
        tag += f"_{target.instance_type.replace('.', '-')}"
    (outdir / f"tuned-{tag}.json").write_text(json.dumps(cfg_doc, indent=2))
    (outdir / f"report-{tag}.md").write_text(
        markdown_report(target, measurements, objective))

    print()
    print("=" * 68)
    print(f"BEST ({objective.name}): {top.config.label()}")
    print(f"  prefill {top.prefill_tps:.1f} tok/s | decode {top.decode_tps:.1f} tok/s"
          f" | TTFT {top.ttft_ms:.0f} ms")
    if base and base.decode_tps:
        print(f"  vs stock all-core default: "
              f"{top.decode_tps / base.decode_tps:.2f}x decode")
    print(f"  run with: {cfg_doc['llama_cli_args']}")
    print("=" * 68)
    print(f"\nWrote {outdir}/tuned-{tag}.json, report-{tag}.md, sweep.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gravitune",
                                description="Autotune llama.cpp inference on Arm.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="identify the Arm target and its kernel paths")
    d.set_defaults(func=cmd_detect)

    t = sub.add_parser("tune", help="sweep configs and emit a tuned config")
    t.add_argument("--bench", required=True, help="path to llama-bench")
    t.add_argument("--models", required=True, nargs="+",
                   help="GGUF model(s); the first is the primary")
    t.add_argument("--objective", default=DEFAULT_OBJECTIVE, choices=sorted(OBJECTIVES),
                   help="what to optimise for")
    t.add_argument("--outdir", default="results")
    t.add_argument("--prompt-tokens", type=int, default=512)
    t.add_argument("--gen-tokens", type=int, default=128)
    t.add_argument("--reps", type=int, default=2)
    t.add_argument("--quick", action="store_true",
                   help="threads only; skip secondary knobs")
    t.set_defaults(func=cmd_tune)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
