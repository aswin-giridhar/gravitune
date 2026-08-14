# Where does the flash-attention result stop holding?

Two follow-ups that test the boundaries of GraviTune's headline finding, because
a claim with untested edges is a claim you cannot defend.

---

## 1. Context length: the effect gets *stronger*, not weaker

The README originally carried this caveat:

> Attention working set grows quadratically with context, so the V-series
> flash-attention result should reverse at long enough context.

**That prediction was wrong.** Measured on `c8g.4xlarge` (Graviton4 / Neoverse-V2),
Qwen2.5-1.5B Q4_0, 16 threads, prefill throughput in tokens/sec:

| prompt tokens | `-fa on` | `-fa off` | `-fa off` advantage |
|---:|---:|---:|---:|
| 512 | 433.2 | 851.8 | 1.97× |
| 1,024 | 282.2 | 745.8 | 2.64× |
| 2,048 | 182.1 | 628.9 | 3.45× |
| 4,096 | 105.9 | 456.4 | 4.31× |
| 8,192 | 57.8 | 300.4 | **5.20×** |

The advantage grows **monotonically**. There is no crossover through 8K context —
disabling flash attention goes from a 2× win to a 5× win.

### Why the prediction was wrong

The reasoning behind the caveat was that attention cost grows quadratically while
the GEMM work grows linearly, so attention should eventually dominate and
FlashAttention's memory-traffic savings should start paying off.

The first half is right; the conclusion does not follow. As context grows,
attention takes a larger share of total time — which means the *penalty* for
running an inefficient attention kernel grows too. On Neoverse-V2 the tiled
flash-attention path costs roughly 2× the instructions of the direct path
([evidence](why-flash-attention-loses.md)). Making attention a bigger fraction of
the workload therefore amplifies a 2× penalty on a growing share of the total,
rather than unlocking a saving.

FlashAttention would only start winning once the working set genuinely exceeds
the cache hierarchy and the kernel becomes bandwidth-bound rather than
compute-bound. On a Neoverse-V2 with its large private L2 and shared SLC, an 8K
context on a 1.5B model has not reached that point. A much longer context, a
much larger model, or a machine with far less cache per core could still get
there — we simply have not found it, and we no longer assert that it exists.

**Practical takeaway: on Graviton4, the longer your prompts, the more `-fa off`
is worth.** Retrieval-augmented and long-document workloads benefit most, which
is the opposite of what the original caveat implied.

---

## 2. x86 comparison: the effect is architecture-specific

Same model, same flags, on an Intel i7-10870H (Comet Lake, AVX2, 8 physical
cores), 512-token prefill:

| | `-fa on` | `-fa off` | effect of disabling |
|---|---:|---:|---|
| Intel i7-10870H (AVX2) | 96.1 | 91.6 | **−4.7% — slightly worse** |
| Graviton4 (Neoverse-V2) | 433.2 | 851.8 | **+97% — much better** |

On x86, flash attention is neutral-to-slightly-positive: the default is fine, and
turning it off is a small loss. On Neoverse-V2 the same flag is worth roughly 2×.

**This is why the finding needed to be made on Arm.** A developer tuning on an
x86 laptop and deploying to Graviton would carry over a setting that is correct
where they tested it and badly wrong where it runs — which is precisely the class
of mistake GraviTune exists to catch.

### What this comparison does NOT show

We are **not** claiming Arm is ~9× faster than x86 from the 851.8 vs 96.1 numbers,
and nobody should read them that way. The two machines differ in core count
(16 vs 8), class (server vs 2020 laptop), sustained power and thermal headroom,
memory bandwidth, and available ISA (Neoverse-V2 has i8mm/SVE2; this Intel part
has AVX2 and no AVX-512 or AMX). A throughput ratio across those differences is
not a meaningful architectural comparison.

What *is* comparable is the **direction and sign of the flash-attention effect**,
because that is measured within each machine against itself. That result is
robust: negative on x86, strongly positive on Neoverse-V2.

---

## 3. Larger model

A 7B sweep was launched alongside these runs to test whether the effect survives
at a model size where the workload is more bandwidth-bound. **It did not finish
before the submission deadline**, so no 7B claim is made here. The 7B result
would be the next thing to check, and `scripts/reproduce.sh` will run it on any
Graviton box.

---

## Reproducing

```bash
for P in 512 1024 2048 4096 8192; do
  for FA in on off; do
    llama-bench -m qwen2.5-1.5b-instruct-q4_0.gguf -t 16 -p $P -n 0 -fa $FA -r 1
  done
done
```

Raw output: [`results/crossover-c8g-4xlarge.txt`](../results/crossover-c8g-4xlarge.txt)
