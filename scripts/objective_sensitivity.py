#!/usr/bin/env python3
"""Is your objective function measuring anything, or just decorating a guess?

Run this against a sweep.json to see how sensitive the winning config is to the
constants inside your scoring function. If the winner flips right next to the
value you chose, your "recommendation" is noise.

    python3 scripts/objective_sensitivity.py results/sweep.json

This exists because GraviTune shipped exactly that bug. The original
`interactive` objective was

    decode_tps * (1 + w * 1000/ttft_ms)     with w = 0.15 chosen by feel

and the winning config flips at w = 0.1469 -- 0.003 away. Since decode
throughput varies ~15% run-to-run on shared cloud hosts, the recommendation was
a coin toss, and nothing in the output would have said so. It is now a TTFT
budget instead (see gravitune/score.py).
"""
import json
import sys
from pathlib import Path


def label(m):
    c = m["config"]
    model = Path(c["model"]).stem.replace("qwen2.5-1.5b-instruct-", "")
    return f"t{c['threads']}_ub{c['ubatch']}_fa{c['flash_attn']}_{model}"


def main(path):
    ms = [m for m in json.loads(Path(path).read_text())["measurements"] if m["ok"]]
    if not ms:
        print("no successful measurements")
        return 1

    def blended(m, w):
        return m["decode_tps"] * (1.0 + w * (1000.0 / (m["ttft_ms"] or 1e9)))

    def winner(w):
        return label(max(ms, key=lambda m: blended(m, w)))

    print("=" * 70)
    print("Weighted-blend objective: winner as a function of the TTFT weight w")
    print("=" * 70)
    prev, flips = None, []
    for i in range(0, 61):
        w = i / 20.0
        n = winner(w)
        if n != prev:
            flips.append((w, n))
            print(f"  w >= {w:4.2f}   ->  {n}")
            prev = n

    if len(flips) == 1:
        print("\n  Stable across the whole range: the weight does no work at all.")
        return 0

    # bisect each transition to find how sharp it is
    print("\n" + "=" * 70)
    print("Flip points (bisected)")
    print("=" * 70)
    for k in range(1, len(flips)):
        lo, hi = flips[k - 1][0], flips[k][0]
        target = flips[k - 1][1]
        for _ in range(60):
            mid = (lo + hi) / 2
            if winner(mid) == target:
                lo = mid
            else:
                hi = mid
        print(f"  {flips[k-1][1]}  ->  {flips[k][1]}")
        print(f"    flips at w = {hi:.6f}")
        print(f"    distance from the shipped 0.15: {abs(0.15 - hi):.6f}")

    print("\n" + "=" * 70)
    print("Budget objective (what GraviTune uses now)")
    print("=" * 70)
    for budget in (400, 600, 800, 1000, 1500, 3000):
        meets = [m for m in ms if m["ttft_ms"] and m["ttft_ms"] <= budget]
        if meets:
            b = max(meets, key=lambda m: m["decode_tps"])
            print(f"  budget {budget:5d} ms -> {label(b):<42} "
                  f"decode {b['decode_tps']:6.1f}  ttft {b['ttft_ms']:5.0f}")
        else:
            b = min(ms, key=lambda m: m["ttft_ms"] or 1e9)
            print(f"  budget {budget:5d} ms -> (none meet it) closest: {label(b)}")
    print("\n  A budget is interpretable: you can state it, defend it, and see")
    print("  exactly which configs qualify. A blend weight cannot be defended,")
    print("  because there is no real exchange rate between ms and tokens/sec.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results/sweep.json"))
