# GraviTune report — Neoverse-N1

- **Target**: Neoverse-N1 (MIDR 0xd0c), 16 cores, `c6g.4xlarge`
- **ISA features**: `asimd asimddp`
- **Objective**: `interactive` — Chat / assistant workloads. Weights decode throughput heavily, since that is the streaming speed a user perceives, with a modest pull toward low TTFT so responses start promptly.
- **Kernel**: 6.17.0-1019-aws

## Recommendation

```
16 threads, batch 2048, ubatch 512, flash-attn on
```

**1.01x decode throughput** vs the stock all-core default (101.8 -> 103.3 tok/s).

## Full sweep

| rank | config | prefill tok/s | decode tok/s | TTFT ms | score |
|---:|---|---:|---:|---:|---:|
| 1 | `t16_b2048_ub512_faon_qwen2.5-1.5b-instruct-q4_0` | 454.2 | 103.3 | 1127 | 117.0 |
| 2 | `t16_b2048_ub1024_faauto_qwen2.5-1.5b-instruct-q4_0` | 452.9 | 102.1 | 1131 | 115.7 |
| 3 | `t16_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_0` | 450.2 | 101.8 | 1137 | 115.2 |
| 4 | `t16_b2048_ub512_faoff_qwen2.5-1.5b-instruct-q4_0` | 440.9 | 96.7 | 1161 | 109.1 |
| 5 | `t16_b2048_ub256_faauto_qwen2.5-1.5b-instruct-q4_0` | 448.8 | 92.6 | 1141 | 104.7 |
| 6 | `t16_b2048_ub128_faauto_qwen2.5-1.5b-instruct-q4_0` | 389.8 | 91.6 | 1313 | 102.0 |
| 7 | `t16_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_k_m` | 267.4 | 89.3 | 1915 | 96.3 |
| 8 | `t16_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q8_0` | 382.6 | 76.8 | 1338 | 85.4 |
| 9 | `t8_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_0` | 240.0 | 67.6 | 2133 | 72.3 |
| 10 | `t4_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_0` | 121.0 | 38.7 | 4230 | 40.0 |
| 11 | `t2_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_0` | 60.5 | 20.5 | 8460 | 20.8 |
| 12 | `t32_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_0` | 420.7 | 17.3 | 1217 | 19.4 |
| 13 | `t1_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_0` | 30.8 | 10.9 | 16647 | 11.0 |
