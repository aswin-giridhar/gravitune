# Why flash attention loses on Graviton4

GraviTune's sweep found that llama.cpp's default `-fa auto` (which resolves to *on*)
costs **1.99× prefill throughput** on Neoverse-V2. This is the hardware evidence for
why, collected with `perf` on a real `c8g.4xlarge`.

## The measurement

Workload: `llama-bench -m qwen2.5-1.5b-instruct-q4_0.gguf -t 16 -p 512 -n 0 -r 1`
(prefill only, 16 threads, 512-token prompt).

| | instructions | cycles | IPC | prefill tok/s |
|---|---:|---:|---:|---:|
| `-fa on` | 389,683,137,185 | 95,608,334,834 | **4.08** | 430.4 |
| `-fa off` | 197,945,396,973 | 56,700,776,193 | **3.49** | 854.2 |
| ratio | **1.97×** | 1.69× | 0.86× | 0.50× |

Flash attention executes **1.97× more instructions to do the same work.** The
instruction ratio (1.97×) tracks the throughput ratio (1.99×) almost exactly, which is
what makes this a causal explanation rather than a correlation.

## The trap: higher IPC, worse performance

Note that `-fa on` has the **better** IPC — 4.08 versus 3.49. An optimization effort
driven by IPC alone would rank the flash-attention path as the more efficient one and
try to make the *other* path more like it.

That would be exactly backwards. IPC measures how efficiently the core retires the
instructions it is given. It says nothing about whether those instructions needed to
exist. Flash attention here is efficiently executing redundant arithmetic — a very
well-pipelined waste of time.

**This is the argument for pairing a sweep with a profiler.** The sweep measures the
outcome the user cares about; the profiler explains it. Either alone misleads.

## Where the time goes

`perf record` on the same two configs, top symbols by self time:

**`-fa on`**
```
54.18%  ggml_compute_forward_flash_attn_ext_tiled
38.95%  ggml_gemm_q4_0_4x8_q8_0
```

**`-fa off`**
```
65.40%  ggml_gemm_q4_0_4x8_q8_0
16.49%  ggml_vec_dot_f16
 2.09%  ggml_compute_forward_soft_max
```

With FA enabled, **over half the machine goes into the tiled flash-attention kernel**,
and the quantized GEMM — the work that actually has to happen — is squeezed into 39%.
Disable it and that GEMM gets 65% of the machine.

`ggml_gemm_q4_0_4x8_q8_0` is the **repacked Q4_0 4×8 kernel**, the int8-matmul-optimized
path. Its dominance confirms the i8mm/dot-product accelerated kernels are genuinely
engaged on this target — which is also what `gravitune detect` predicts from the ISA
flags before you run anything.

## The underlying reason

FlashAttention is a GPU-motivated algorithm. Its central trade is **more arithmetic in
exchange for fewer round-trips to memory**, because on a GPU the tiny on-chip SRAM
scratchpad is the binding constraint, and HBM round-trips dominate attention cost.

A Neoverse-V2 core has a large private L2 and sits behind a big shared L3/system-level
cache. For a 1.5B model at a 512-token prompt, the attention working set already fits
comfortably in cache. **The memory traffic flash attention exists to avoid was never the
bottleneck.** So the trade is all cost and no benefit: you pay the extra arithmetic and
save nothing.

This is a general lesson rather than a llama.cpp bug: an algorithm that is optimal under
one memory hierarchy can be pessimal under another, and the default carried over
unchanged.

## Caveats

- Specific to **CPU** inference. Do not carry this to a GPU backend, where FA is a large
  win for exactly the reasons above.
- Measured at 512-token prompts on a 1.5B model. As context length grows the attention
  working set grows quadratically and will eventually exceed cache — at which point FA's
  trade starts to pay off. **Where that crossover sits is the obvious next experiment**,
  and GraviTune's sweep is the right tool to find it.
- Decode (`-n`) is affected much less than prefill; the win here is a prefill/TTFT win.

## Reproducing

```bash
perf stat -e cycles,instructions -- \
  llama-bench -m model.gguf -t 16 -p 512 -n 0 -fa on  -r 1
perf stat -e cycles,instructions -- \
  llama-bench -m model.gguf -t 16 -p 512 -n 0 -fa off -r 1
```

Ubuntu ships `kernel.perf_event_paranoid = 4`, which blocks unprivileged `perf`. Use
`sudo`, or `sudo sysctl -w kernel.perf_event_paranoid=-1`. That is a distro hardening
default, **not** an AWS restriction — Graviton VMs do pass the PMU through.
