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

# Default TTFT budget in milliseconds.
#
# 1000 ms is the long-standing HCI threshold for keeping a user's "flow of
# thought" uninterrupted (Nielsen, *Usability Engineering*). We set the budget
# slightly under it so there is headroom for network and template overhead that
# llama-bench does not measure.
TTFT_BUDGET_MS = 800.0


def _interactive(m: Measurement) -> float:
    """Maximise decode throughput subject to a time-to-first-token budget.

    An earlier version blended the two into a single weighted score:

        decode_tps * (1 + w * 1000/ttft_ms)      with w = 0.15

    That was wrong, and measurably so. Sweeping `w` over the real sweep data
    showed the winning config flips at w = 0.1469 -- 0.003 away from the 0.15
    that had been chosen by feel. Since decode throughput varies ~15%
    run-to-run on shared cloud hosts, the "recommendation" was a coin toss
    balanced on a knife edge, and nothing in the output would have revealed
    that. `scripts/objective_sensitivity.py` reproduces the finding.

    A budget is the honest formulation because it states the actual
    requirement: a response must *start* within a tolerable delay, and after
    that the thing users feel is streaming speed. Configs that meet the budget
    are ranked purely on decode; configs that miss it are ranked below all of
    them, ordered by how close they came. There is no hidden exchange rate
    between milliseconds and tokens per second, because no such rate exists.
    """
    if m.ttft_ms and m.ttft_ms <= TTFT_BUDGET_MS:
        return 1e6 + m.decode_tps          # meets budget: rank on decode alone
    return -_safe(m.ttft_ms)               # misses budget: least-bad first


INTERACTIVE = Objective(
    name="interactive",
    description=(
        f"Chat / assistant workloads. Maximises decode throughput subject to a "
        f"time-to-first-token budget of {TTFT_BUDGET_MS:.0f} ms, rather than "
        f"trading the two off against each other at an arbitrary exchange rate."
    ),
    score=_interactive,
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


# Published on-demand USD/hour, us-west-2, at time of measurement. Used only to
# express throughput per unit cost; override with --price-per-hour for your own
# region, instance size, or negotiated/spot rate.
DEFAULT_PRICE_PER_HOUR = {
    "c8g.4xlarge": 0.72666,
    "c7g.4xlarge": 0.58,
    "c6g.4xlarge": 0.544,
}


def cost_objective(price_per_hour: float) -> Objective:
    """Tokens per dollar.

    On Arm the interesting question is usually not "which box is fastest" but
    "which box does the work for less". A config that gives up 8% throughput on
    hardware costing 25% less wins on the invoice every month, and that is the
    comparison an infrastructure owner actually makes.

    Rate is a caller-supplied price because there is no correct universal
    number: region, instance size, spot vs on-demand, and committed-use
    discounts all move it. Passing it in keeps the arithmetic honest rather
    than baking a stale list price into a score.
    """
    return Objective(
        name="cost",
        description=(
            f"Decode tokens per US dollar at ${price_per_hour:.4f}/hour. "
            f"Optimises throughput per unit spend rather than raw speed."
        ),
        # tok/s * 3600 s/hr / ($/hr) = tokens per dollar
        score=lambda m: (m.decode_tps * 3600.0) / _safe(price_per_hour),
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
