"""Selecting the 'best' config from a sweep.

This is the one genuinely opinionated part of the tool. Everything else measures;
this decides. There is no universally correct answer -- the right config depends
on what the deployment is optimising for -- so the objective is a first-class,
swappable thing rather than a hardcoded `max(decode_tps)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .sweep import Measurement


@dataclass(frozen=True)
class Objective:
    """A named way of ranking configs."""

    name: str
    description: str
    score: Callable[[Measurement], float]


def _safe(x: float) -> float:
    return x if x and x > 0 else 1e-9


# --- Built-in objectives ---------------------------------------------------
#
# Why these three: prefill and decode pull in *opposite* directions on Arm
# servers. Prefill (prompt processing) is a compute-bound GEMM that likes lots
# of threads and big batches. Decode (token generation) is bandwidth-bound and
# barrier-heavy, so it degrades badly under oversubscription. A config tuned for
# one can be actively bad for the other -- which is exactly why picking an
# objective has to be a deliberate choice.

INTERACTIVE = Objective(
    name="interactive",
    description=(
        "Chat / assistant workloads. Weights decode throughput heavily, since "
        "that is the streaming speed a user perceives, with a modest pull "
        "toward low TTFT so responses start promptly."
    ),
    score=lambda m: m.decode_tps * (1.0 + 0.15 * (1000.0 / _safe(m.ttft_ms))),
)

BATCH = Objective(
    name="batch",
    description=(
        "Offline/bulk processing: summarisation over a corpus, embeddings, "
        "evaluation runs. Total tokens pushed is what matters; nobody is "
        "watching a cursor blink, so TTFT is ignored."
    ),
    score=lambda m: m.prefill_tps,
)

BALANCED = Objective(
    name="balanced",
    description=(
        "Mixed serving where both long prompts and long generations occur. "
        "Harmonic mean punishes configs that sacrifice one phase entirely, "
        "which a plain average would happily hide."
    ),
    score=lambda m: 2.0 / (1.0 / _safe(m.prefill_tps) + 1.0 / _safe(m.decode_tps)),
)


# ---------------------------------------------------------------------------
# TODO(user): add your own objective here.
#
# This is the piece worth your judgement rather than mine. The three above cover
# throughput-shaped goals, but real deployments are often constrained by
# something else entirely, and that constraint should drive config selection:
#
#   - Cost-efficiency: tokens per dollar. A 16-thread config that is 8% slower
#     than 32 threads but runs on a box costing half as much wins on every
#     invoice. You would divide throughput by an instance $/hr passed in.
#   - Tail latency: p99 TTFT rather than the mean, for anything user-facing
#     under an SLA. Needs Measurement to carry a distribution, not just a mean.
#   - Energy: tokens per watt, which is arguably the entire point of Arm in the
#     datacentre and would need an RAPL-equivalent counter reading.
#
# Signature to match: score(m: Measurement) -> float, higher is better.
#
# CUSTOM = Objective(
#     name="...",
#     description="...",
#     score=lambda m: ...,
# )
# ---------------------------------------------------------------------------


OBJECTIVES = {o.name: o for o in (INTERACTIVE, BATCH, BALANCED)}
DEFAULT_OBJECTIVE = "interactive"


def rank(measurements: Sequence[Measurement],
         objective: Objective) -> list[tuple[Measurement, float]]:
    """Rank successful measurements best-first under `objective`.

    Failed measurements are dropped rather than scored as zero: a config that
    crashed is not 'the slowest config', it is an absence of data, and letting
    it sit at the bottom of a ranking implies we learned something we did not.
    """
    scored = [(m, objective.score(m)) for m in measurements if m.ok]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


def best(measurements: Sequence[Measurement], objective: Objective) -> Measurement | None:
    ranked = rank(measurements, objective)
    return ranked[0][0] if ranked else None


def baseline(measurements: Sequence[Measurement], cores: int) -> Measurement | None:
    """The config a developer would plausibly have used without this tool.

    Defined as llama.cpp's own default threading (all cores) at stock batch
    settings. The speedup we claim is measured against this, not against an
    artificially crippled single-thread run -- overstating the baseline is the
    easiest way to manufacture an impressive-looking number.
    """
    for m in measurements:
        c = m.config
        if (m.ok and c.threads == cores and c.ubatch == 512
                and c.batch == 2048 and c.flash_attn == "auto"):
            return m
    return None
