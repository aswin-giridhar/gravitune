"""Sweep engine: measure llama.cpp inference across a config grid on Arm.

Stdlib only, on purpose. This is meant to run on a freshly-provisioned Arm box
where every `pip install` is a chance for a missing aarch64 wheel to turn into a
slow source build.

The unit of work is a Config -> one llama-bench invocation -> a Measurement.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator


# --------------------------------------------------------------------------
# Arm target identification
# --------------------------------------------------------------------------

# MIDR part numbers -> human names. Arm encodes the CPU part in bits [15:4] of
# MIDR_EL1, which is what actually distinguishes Graviton generations. Instance
# type strings ("c7g") are an AWS-ism and tell you nothing on Azure/GCP/bare metal.
_MIDR_PARTS = {
    0xD0C: "Neoverse-N1",   # Graviton2, Ampere Altra
    0xD40: "Neoverse-V1",   # Graviton3
    0xD49: "Neoverse-N2",   # Microsoft Cobalt 100
    0xD4F: "Neoverse-V2",   # Graviton4, NVIDIA Grace, GCP Axion
    0xD8E: "Neoverse-N3",
    0xD84: "Neoverse-V3",
}

# ISA features that gate which llama.cpp / KleidiAI kernel path is reachable.
# These are the difference between "fast" and "silently falls back".
_FEATURES_OF_INTEREST = [
    "asimd", "asimddp", "i8mm", "bf16", "sve", "sve2", "svei8mm", "svebf16", "sme",
]


@dataclass
class Target:
    """What Arm machine are we actually on."""

    cpu_part: str = "unknown"
    midr: str = ""
    cores: int = 0
    features: list[str] = field(default_factory=list)
    kernel: str = ""
    instance_type: str = ""

    @property
    def has_i8mm(self) -> bool:
        return "i8mm" in self.features

    @property
    def has_sve(self) -> bool:
        return "sve" in self.features


def detect_target() -> Target:
    """Identify the Arm CPU from MIDR + /proc/cpuinfo, not from a cloud label."""
    t = Target(kernel=platform.release())

    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        return t

    t.cores = cpuinfo.count("processor\t:")

    m = re.search(r"CPU part\s*:\s*(0x[0-9a-fA-F]+)", cpuinfo)
    if m:
        part = int(m.group(1), 16)
        t.midr = m.group(1)
        t.cpu_part = _MIDR_PARTS.get(part, f"unknown-arm-{m.group(1)}")

    m = re.search(r"Features\s*:\s*(.+)", cpuinfo)
    if m:
        present = set(m.group(1).split())
        t.features = [f for f in _FEATURES_OF_INTEREST if f in present]

    # Cloud instance type, when we happen to be on a cloud box. Best-effort:
    # absence is normal (bare metal), not an error.
    try:
        tok = subprocess.run(
            ["curl", "-sS", "-m", "1", "-X", "PUT",
             "http://169.254.169.254/latest/api/token",
             "-H", "X-aws-ec2-metadata-token-ttl-seconds: 60"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        if tok:
            t.instance_type = subprocess.run(
                ["curl", "-sS", "-m", "1",
                 "http://169.254.169.254/latest/meta-data/instance-type",
                 "-H", f"X-aws-ec2-metadata-token: {tok}"],
                capture_output=True, text=True, timeout=3,
            ).stdout.strip()
    except Exception:
        pass

    return t


# --------------------------------------------------------------------------
# Config grid
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """One point in the tuning space."""

    threads: int
    model: str
    batch: int = 2048
    ubatch: int = 512
    flash_attn: str = "auto"

    def label(self) -> str:
        return (f"t{self.threads}_b{self.batch}_ub{self.ubatch}"
                f"_fa{self.flash_attn}_{Path(self.model).stem}")


@dataclass
class Measurement:
    """Result of running one Config."""

    config: Config
    prefill_tps: float = 0.0     # pp: prompt processing, compute-bound
    decode_tps: float = 0.0      # tg: token generation, bandwidth/sync-bound
    ttft_ms: float = 0.0         # derived: time to first token for a 512-tok prompt
    ok: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["config"] = asdict(self.config)
        return d


def thread_candidates(cores: int) -> list[int]:
    """Thread counts worth testing on an Arm server.

    Deliberately includes an oversubscribed point (2x cores). Arm server cores
    have no SMT, so oversubscription is a trap that x86 habits walk straight
    into -- and we can only *show* that if we measure it.
    """
    cands = {1, 2, 4, cores // 4, cores // 2, cores, cores * 2}
    return sorted(c for c in cands if c >= 1)


def build_grid(models: list[str], cores: int, quick: bool = False) -> list[Config]:
    """Full factorial would be wasteful. Sweep threads densely (it dominates),
    then vary the secondary knobs only at the best-guess thread count."""
    grid: list[Config] = []
    primary = models[0]

    for t in thread_candidates(cores):
        grid.append(Config(threads=t, model=primary))

    if quick:
        return grid

    # Secondary knobs, held at full-core threading.
    for ub in (128, 256, 1024):
        grid.append(Config(threads=cores, model=primary, ubatch=ub))
    for fa in ("on", "off"):
        grid.append(Config(threads=cores, model=primary, flash_attn=fa))

    # Quant format comparison -- the KleidiAI dispatch story lives here.
    for m in models[1:]:
        grid.append(Config(threads=cores, model=m))

    return grid


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

class SweepRunner:
    def __init__(self, llama_bench: str, prompt_tokens: int = 512,
                 gen_tokens: int = 128, reps: int = 2):
        self.llama_bench = llama_bench
        self.prompt_tokens = prompt_tokens
        self.gen_tokens = gen_tokens
        self.reps = reps

    def _cmd(self, cfg: Config) -> list[str]:
        return [
            self.llama_bench,
            "-m", cfg.model,
            "-t", str(cfg.threads),
            "-p", str(self.prompt_tokens),
            "-n", str(self.gen_tokens),
            "-b", str(cfg.batch),
            "-ub", str(cfg.ubatch),
            "-fa", cfg.flash_attn,
            "-r", str(self.reps),
            "-o", "json",
        ]

    def run_one(self, cfg: Config, timeout: int = 900) -> Measurement:
        try:
            proc = subprocess.run(
                self._cmd(cfg), capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return Measurement(cfg, ok=False, error="timeout")

        if proc.returncode != 0:
            # Keep the real reason. A config that fails to run and a config that
            # runs slowly must never look the same downstream.
            tail = (proc.stderr or "").strip().splitlines()
            return Measurement(cfg, ok=False,
                               error=f"exit {proc.returncode}: {tail[-1] if tail else '?'}")

        return self._parse(cfg, proc.stdout)

    def _parse(self, cfg: Config, stdout: str) -> Measurement:
        try:
            rows = json.loads(stdout)
        except json.JSONDecodeError as e:
            return Measurement(cfg, ok=False, error=f"unparseable json: {e}")

        m = Measurement(cfg)
        for row in rows:
            kind = row.get("n_prompt", 0), row.get("n_gen", 0)
            tps = float(row.get("avg_ts", 0.0))
            if kind[0] and not kind[1]:
                m.prefill_tps = tps
            elif kind[1] and not kind[0]:
                m.decode_tps = tps

        if m.prefill_tps > 0:
            # TTFT for a prompt of prompt_tokens, in ms. This is the number a
            # user of an interactive app actually perceives.
            m.ttft_ms = (self.prompt_tokens / m.prefill_tps) * 1000.0

        if m.prefill_tps == 0 and m.decode_tps == 0:
            m.ok = False
            m.error = "no throughput rows in output"

        return m

    def sweep(self, grid: list[Config]) -> Iterator[Measurement]:
        for i, cfg in enumerate(grid, 1):
            started = time.time()
            meas = self.run_one(cfg)
            elapsed = time.time() - started
            status = "ok" if meas.ok else f"FAIL({meas.error})"
            print(f"[{i}/{len(grid)}] {cfg.label()}: "
                  f"prefill={meas.prefill_tps:.1f} decode={meas.decode_tps:.1f} "
                  f"ttft={meas.ttft_ms:.0f}ms [{status}] ({elapsed:.0f}s)",
                  flush=True)
            yield meas
