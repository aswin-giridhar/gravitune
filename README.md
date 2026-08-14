# GraviTune

[![Arm CI](https://github.com/aswin-giridhar/gravitune/actions/workflows/arm-ci.yml/badge.svg)](https://github.com/aswin-giridhar/gravitune/actions/workflows/arm-ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Every result here is verified on **real Arm silicon in CI** — the badge above runs
> GraviTune on GitHub's free `ubuntu-24.04-arm` runner (image managed by Arm, LLC) on
> every push. You can confirm it works without provisioning any hardware.

**Autotune LLM inference on Arm — and find out that the defaults are wrong.**

GraviTune measures llama.cpp inference across a config grid on the Arm machine you
are actually on, picks the best config for a stated objective, and emits a reusable
tuned-config artifact with the evidence attached.

It was built for the Arm Create: AI Optimization Challenge (Cloud AI track) and every
number below was measured on real AWS Graviton silicon — no emulation, no estimates.

---

## The headline result

On **AWS Graviton4** (`c8g.4xlarge`, Neoverse-V2, 16 cores), running
Qwen2.5-1.5B-Instruct Q4_0, GraviTune found two things that cost real users real time:

### 1. llama.cpp's default flash-attention setting halves your prefill speed

| config | prefill tok/s | TTFT (512-tok prompt) |
|---|---:|---:|
| `-fa auto` ← **the default** | 430.4 | 1190 ms |
| `-fa off` | **854.2** | **599 ms** |

**1.99× prefill throughput. 1.99× faster time-to-first-token. From one flag.**

Reproduced independently on a clean instance via `scripts/reproduce.sh`: 853.8 tok/s,
600 ms — within 0.05%.

**We deliberately do not claim a decode win.** Decode measured 127.5 tok/s and 146.8
tok/s for the *same* baseline config across two runs — a 15% run-to-run swing, so any
decode delta here is inside the noise floor. Prefill saturates compute and is highly
reproducible; decode is bandwidth-bound and sensitive to memory-system state and
neighbour activity on a shared host. **The honest result is a prefill/TTFT win.**

FlashAttention is a GPU-motivated algorithm: it spends extra arithmetic to avoid
round-trips to memory, because on a GPU the small SRAM scratchpad is the binding
constraint. A Neoverse-V2 core has a large private L2 and a big shared L3/SLC, so the
attention working set already lives in cache. The memory traffic FA exists to avoid was
never the bottleneck here — you pay the extra arithmetic and get nothing back.

`-fa auto` resolves to *on*, so **the default is the slow path on Neoverse V-series cores.**

**But not on every Arm core** — and that is the point of measuring:

| prefill, 16t, Q4_0 | `-fa auto` | `-fa off` | ratio |
|---|---:|---:|---:|
| Graviton4 `c8g` (Neoverse-V2) | 430.4 | 854.2 | **1.99x** |
| Graviton3 `c7g` (Neoverse-V1) | 397.9 | 615.4 | **1.55x** |
| Graviton2 `c6g` (Neoverse-N1) | 450.2 | 440.9 | 0.98x - no win |

The split tracks the **core family, not the generation number**. Neoverse V-series are
wide performance cores with large caches, so FA's trade buys nothing. Neoverse-N1 is a
narrower efficiency core with a smaller cache hierarchy, where the memory traffic FA
avoids is real and the default is correct.

If the optimal config were the same everywhere you would not need an autotuner, you
would need a blog post. See [docs/generational-comparison.md](docs/generational-comparison.md).

### 2. Oversubscribing threads costs 4.5× on token generation

| threads (16 physical cores) | prefill tok/s | decode tok/s |
|---|---:|---:|
| 16 | 430.4 | 127.5 |
| 32 | 417.5 | **28.1** |

Prefill barely moves; decode falls off a cliff. Prefill is a compute-bound GEMM, so
oversubscription just adds scheduler overhead. Decode is a bandwidth-bound GEMV with a
**barrier across every thread on every token** — and Arm server cores have **no SMT**,
so each barrier now waits on a descheduled thread while spin-waiters burn the cores
their peers need.

On x86, `nproc` reports 32 on a 16-core SMT box and habitually oversubscribing is often
harmless. That reflex carried onto Arm is a 4.5× penalty on the metric users feel.

---

## Why this matters

Both findings share a shape: **a default that is correct somewhere else.** One is a
GPU-era algorithm applied to a CPU, the other an x86 SMT habit applied to a
no-SMT core. Neither is visible from a model card, a README, or an x86 dev box. They
only appear when you measure on the actual silicon — which is exactly what GraviTune
automates so that each developer doesn't have to rediscover them.

---

## Install and run

No dependencies beyond Python 3.10+ and a llama.cpp build. GraviTune is stdlib-only on
purpose: on a fresh Arm box, every `pip install` is a chance for a missing `aarch64`
wheel to become a slow source build.

```bash
# 1. Get llama.cpp (skip if you have it)
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=ON
cmake --build build -j"$(nproc)" && cd ..

# 2. Get GraviTune
git clone https://github.com/aswin-giridhar/gravitune
cd gravitune

# 3. What Arm chip is this, really?
python3 -m gravitune detect

# 4. Autotune
curl -sSL -o model.gguf \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_0.gguf

python3 -m gravitune tune \
  --bench ../llama.cpp/build/bin/llama-bench \
  --models model.gguf \
  --objective interactive
```

`detect` on a Graviton4 prints:

```
CPU part      : Neoverse-V2 (MIDR 0xd4f)
Cores         : 16
Instance type : c8g.4xlarge
Kernel        : 6.17.0-1019-aws
ISA features  : asimd asimddp i8mm bf16 sve sve2 svei8mm svebf16

Kernel dispatch implications:
  i8mm (int8 matmul, KleidiAI int4/int8 GEMM) : YES
  SVE                                        : YES
```

Identification is from **MIDR_EL1**, the architectural CPU ID register — not from the
cloud instance label. `c7g` tells you what you were billed for; MIDR tells you which
microarchitecture you got, which is what actually determines the kernel path. The same
detection works unchanged on Cobalt, Axion, Grace, Ampere, or a Raspberry Pi.

### Output

`tune` writes three artifacts to `results/`:

| file | what it is |
|---|---|
| `tuned-<cpu>_<instance>.json` | the reusable artifact — recommended flags + measured evidence + full target provenance |
| `report-<cpu>_<instance>.md` | human-readable report with the full ranked sweep |
| `sweep.json` | every raw measurement, for your own analysis |

Apply the result directly:

```bash
./llama-cli -m model.gguf $(jq -r .llama_cli_args results/tuned-neoverse-v2_c8g-4xlarge.json)
```

---

## Choosing what to optimise for

There is no single "fastest" config: prefill and decode pull in opposite directions.
The objective is therefore explicit and swappable (`gravitune/score.py`):

| objective | for | ranks by |
|---|---|---|
| `interactive` (default) | chat, assistants | decode throughput, with a pull toward low TTFT |
| `batch` | bulk summarisation, embeddings, evals | prefill throughput |
| `balanced` | mixed serving | harmonic mean, so neither phase can be sacrificed |

Add your own by appending an `Objective` — there is a marked `TODO(user)` block in
`score.py` with worked suggestions (tokens-per-dollar, p99 TTFT, tokens-per-watt).

---

## MCP server — let an agent tune the box it is on

The Cloud AI track is about agentic workloads, and this is the piece that plugs in.
GraviTune ships an MCP server (stdio, JSON-RPC 2.0, stdlib only) so a coding agent can
interrogate the actual machine instead of guessing:

```json
{
  "mcpServers": {
    "gravitune": {
      "command": "python3",
      "args": ["-m", "gravitune.mcp_server"],
      "env": { "GRAVITUNE_BENCH": "/path/to/llama.cpp/build/bin/llama-bench" }
    }
  }
}
```

| tool | answers |
|---|---|
| `detect_arm_target` | Which Arm core is this, how many, what ISA? |
| `check_kernel_dispatch` | Will KleidiAI's int4/int8 GEMM kernels engage, or silently fall back? |
| `autotune_inference` | Sweep and return the best config with before/after numbers. |

Verified working on Graviton4 — handshake, `tools/list`, and live tool calls all
returning real hardware data.

---

## Profiling with Arm Performix, and a caveat worth publishing

[Arm Performix](https://developer.arm.com/servers-and-cloud-computing/arm-performix) is
Arm's free performance-analysis toolkit for Neoverse. We used it to explain *why* the
tuned configs win. In the process we hit an undocumented constraint that other
developers will hit too:

**AWS Graviton VMs expose only 2 general-purpose PMU counters to the guest.** The kernel
reports `3 (0,80000003) counters available`, but one is the dedicated cycle counter
(`PMCCNTR`). Raw `perf` accepts 3 events and rejects 4.

| Performix recipe | Graviton VM | why |
|---|---|---|
| `code_hotspots` | ✅ works | sampling-based |
| `system_utilization` | ✅ works | no PMU multiplexing |
| `instruction_mix` | ❌ needs 3+ GP counters | `INSUFFICIENT_PMU_COUNTERS` |
| `cpu_microarchitecture` | ❌ needs 3+ GP counters | `INSUFFICIENT_PMU_COUNTERS` |

**Rule of thumb: on Graviton VMs use sampling recipes; counter-derived microarchitecture
recipes need bare metal (`*.metal`).** Also note Ubuntu ships
`kernel.perf_event_paranoid = 4`, which blocks unprivileged `perf` — that is a distro
default, *not* an AWS restriction. `sudo sysctl -w kernel.perf_event_paranoid=-1` fixes it.

We could find this documented nowhere, so it is written down here.

---

## Reproducing on a fresh Arm box

Everything above was produced by [`scripts/reproduce.sh`](scripts/reproduce.sh), which
runs end-to-end on a clean instance. Launch a Graviton box and run it:

```bash
curl -sSL https://raw.githubusercontent.com/aswin-giridhar/gravitune/master/scripts/reproduce.sh | bash
```

Tested on `c8g.4xlarge`, `c7g.4xlarge`, and `c6g.4xlarge` running Ubuntu 24.04 arm64.

---

## Published tuned configs

Pre-measured configs live in [`configs/`](configs/) so you can skip the sweep entirely
if you are on hardware we have already characterised. See
[`docs/generational-comparison.md`](docs/generational-comparison.md) for how the same
model behaves across Graviton 2 → 3 → 4.

---

## Limitations

Stated plainly, because a benchmark that hides its scope is not much of a benchmark:

- Measured on one model family (Qwen2.5-1.5B-Instruct) at three quantisations. Larger
  models are more bandwidth-bound and will shift the optimal thread count.
- Single-socket, single-NUMA-node instances. NUMA pinning is not yet swept.
- The `-fa off` result is specific to CPU inference. Do not carry it to a GPU backend.
- `llama-bench` reports steady-state throughput; TTFT is derived from prefill rate
  rather than measured on a cold request.
- **Decode throughput varies ~15% run-to-run** on shared cloud instances. Treat single-run
  decode comparisons with suspicion; prefill reproduces to <0.1%. Raise `--reps` and
  repeat the sweep before acting on any small decode difference.

## License

MIT — see [LICENSE](LICENSE).
