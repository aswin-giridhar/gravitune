# Proposed llama.cpp patch

`llamacpp-fa-auto-arm.patch` makes `-fa auto` resolve to *off* on Arm CPUs with
i8mm + SVE (Neoverse V1/V2 — AWS Graviton3/4), where the tiled CPU
flash-attention path costs ~2x prefill throughput. Neoverse-N1 has neither
feature and is correctly unaffected.

**Status: compiled and behaviourally verified on a real `c8g.4xlarge`.**

| run | prefill tok/s (pp256, 16 threads) |
|---|---:|
| explicit `-fa on` | 361.81 |
| explicit `-fa off` | 519.57 |
| `auto`, patched | **515.85** — resolves to off |

Override with `-fa on` still works.

Uses existing public API (`ggml_cpu_has_matmul_int8()`, `ggml_cpu_has_sve()`);
no MIDR parsing needed. Applies to CPU-only inference; GPU backends unaffected.

**Submitted upstream as [PR #27092](https://github.com/ggml-org/llama.cpp/pull/27092).**

Discussion and full data: https://github.com/ggml-org/llama.cpp/issues/27086

```bash
git apply patches/llamacpp-fa-auto-arm.patch   # from a llama.cpp checkout
```
