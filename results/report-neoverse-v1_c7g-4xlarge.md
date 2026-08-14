# GraviTune report — Neoverse-V1

- **Target**: Neoverse-V1 (MIDR 0xd40), 16 cores, `c7g.4xlarge`
- **ISA features**: `asimd asimddp i8mm bf16 sve svei8mm svebf16`
- **Objective**: `interactive` — Chat / assistant workloads. Weights decode throughput heavily, since that is the streaming speed a user perceives, with a modest pull toward low TTFT so responses start promptly.
- **Kernel**: 6.17.0-1019-aws

## Recommendation

```
16 threads, batch 2048, ubatch 512, flash-attn auto
```

**1.00x decode throughput** vs the stock all-core default (138.2 -> 138.2 tok/s).

## Full sweep

| rank | config | prefill tok/s | decode tok/s | TTFT ms | score |
|---:|---|---:|---:|---:|---:|
| 1 | `t16_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_0` | 397.9 | 138.2 | 1287 | 154.3 |
| 2 | `t16_b2048_ub256_faauto_qwen2.5-1.5b-instruct-q4_0` | 378.2 | 136.6 | 1354 | 151.8 |
| 3 | `t16_b2048_ub512_faon_qwen2.5-1.5b-instruct-q4_0` | 397.7 | 135.9 | 1287 | 151.7 |
| 4 | `t16_b2048_ub1024_faauto_qwen2.5-1.5b-instruct-q4_0` | 397.5 | 134.2 | 1288 | 149.8 |
| 5 | `t16_b2048_ub512_faoff_qwen2.5-1.5b-instruct-q4_0` | 615.4 | 126.8 | 832 | 149.7 |
| 6 | `t16_b2048_ub128_faauto_qwen2.5-1.5b-instruct-q4_0` | 228.5 | 137.0 | 2240 | 146.2 |
| 7 | `t16_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_k_m` | 288.6 | 109.0 | 1774 | 118.3 |
| 8 | `t16_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q8_0` | 355.3 | 97.9 | 1441 | 108.1 |
| 9 | `t8_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_0` | 258.4 | 93.9 | 1982 | 101.0 |
| 10 | `t4_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_0` | 147.7 | 56.4 | 3466 | 58.8 |
| 11 | `t2_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_0` | 73.2 | 31.4 | 6994 | 32.1 |
| 12 | `t32_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_0` | 385.7 | 22.7 | 1328 | 25.3 |
| 13 | `t1_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_0` | 39.6 | 16.9 | 12937 | 17.1 |
