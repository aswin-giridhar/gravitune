# GraviTune report — Neoverse-V2

- **Target**: Neoverse-V2 (MIDR 0xd4f), 16 cores, `c8g.4xlarge`
- **ISA features**: `asimd asimddp i8mm bf16 sve sve2 svei8mm svebf16`
- **Objective**: `interactive` — Chat / assistant workloads. Weights decode throughput heavily, since that is the streaming speed a user perceives, with a modest pull toward low TTFT so responses start promptly.
- **Kernel**: 6.17.0-1019-aws

## Recommendation

```
16 threads, batch 2048, ubatch 512, flash-attn off
```

**1.07x decode throughput** vs the stock all-core default (127.5 -> 136.4 tok/s).

## Full sweep

| rank | config | prefill tok/s | decode tok/s | TTFT ms | score |
|---:|---|---:|---:|---:|---:|
| 1 | `t16_b2048_ub512_faoff_qwen2.5-1.5b-instruct-q4_0` | 854.2 | 136.4 | 599 | 170.5 |
| 2 | `t16_b2048_ub1024_faauto_qwen2.5-1.5b-instruct-q4_0` | 429.9 | 151.2 | 1191 | 170.2 |
| 3 | `t16_b2048_ub512_faon_qwen2.5-1.5b-instruct-q4_0` | 430.6 | 150.0 | 1189 | 168.9 |
| 4 | `t16_b2048_ub256_faauto_qwen2.5-1.5b-instruct-q4_0` | 413.0 | 149.5 | 1240 | 167.6 |
| 5 | `t16_b2048_ub128_faauto_qwen2.5-1.5b-instruct-q4_0` | 250.4 | 149.8 | 2045 | 160.8 |
| 6 | `t16_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_k_m` | 316.2 | 133.3 | 1619 | 145.6 |
| 7 | `t16_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_0` | 430.4 | 127.5 | 1190 | 143.6 |
| 8 | `t16_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q8_0` | 395.2 | 120.9 | 1295 | 134.9 |
| 9 | `t8_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_0` | 282.1 | 101.4 | 1815 | 109.7 |
| 10 | `t4_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_0` | 152.9 | 62.5 | 3348 | 65.3 |
| 11 | `t2_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_0` | 75.9 | 34.3 | 6744 | 35.0 |
| 12 | `t32_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_0` | 417.5 | 28.1 | 1226 | 31.5 |
| 13 | `t1_b2048_ub512_faauto_qwen2.5-1.5b-instruct-q4_0` | 40.9 | 18.3 | 12532 | 18.5 |
