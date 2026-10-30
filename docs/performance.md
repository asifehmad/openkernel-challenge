# Performance comparison — all scored configurations

No GPU timings have been measured. Every latency below is deliberately TBD.
Weights and dimensions follow the corroborated task snapshot, pending live verification.
All shapes have multiplicity 62. Output dtype is BF16 except FP32 router.

| Load | Layer | M | N | K | SGLang µs | Candidate µs | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode 1 | qkv_gate | 1 | 2304 | 8192 | TBD | TBD | TBD |
| decode 1 | o | 1 | 8192 | 1024 | TBD | TBD | TBD |
| decode 1 | pooler | 1 | 4096 | 8192 | TBD | TBD | TBD |
| decode 1 | unpooler | 1 | 1024 | 4096 | TBD | TBD | TBD |
| decode 1 | router | 1 | 256 | 8192 | TBD | TBD | TBD |
| decode 16 | qkv_gate | 16 | 2304 | 8192 | TBD | TBD | TBD |
| decode 16 | o | 16 | 8192 | 1024 | TBD | TBD | TBD |
| decode 16 | pooler | 16 | 4096 | 8192 | TBD | TBD | TBD |
| decode 16 | unpooler | 16 | 8192 | 4096 | TBD | TBD | TBD |
| decode 16 | router | 16 | 256 | 8192 | TBD | TBD | TBD |
| decode 32 | qkv_gate | 32 | 2304 | 8192 | TBD | TBD | TBD |
| decode 32 | o | 32 | 8192 | 1024 | TBD | TBD | TBD |
| decode 32 | pooler | 32 | 4096 | 8192 | TBD | TBD | TBD |
| decode 32 | unpooler | 32 | 8192 | 4096 | TBD | TBD | TBD |
| decode 32 | router | 32 | 256 | 8192 | TBD | TBD | TBD |
| verify 1 | qkv_gate | 8 | 2304 | 8192 | TBD | TBD | TBD |
| verify 1 | o | 8 | 8192 | 1024 | TBD | TBD | TBD |
| verify 1 | pooler | 8 | 4096 | 8192 | TBD | TBD | TBD |
| verify 1 | unpooler | 8 | 1024 | 4096 | TBD | TBD | TBD |
| verify 1 | router | 8 | 256 | 8192 | TBD | TBD | TBD |
| verify 16 | qkv_gate | 128 | 2304 | 8192 | TBD | TBD | TBD |
| verify 16 | o | 128 | 8192 | 1024 | TBD | TBD | TBD |
| verify 16 | pooler | 128 | 4096 | 8192 | TBD | TBD | TBD |
| verify 16 | unpooler | 128 | 8192 | 4096 | TBD | TBD | TBD |
| verify 16 | router | 128 | 256 | 8192 | TBD | TBD | TBD |
| verify 32 | qkv_gate | 256 | 2304 | 8192 | TBD | TBD | TBD |
| verify 32 | o | 256 | 8192 | 1024 | TBD | TBD | TBD |
| verify 32 | pooler | 256 | 4096 | 8192 | TBD | TBD | TBD |
| verify 32 | unpooler | 256 | 8192 | 4096 | TBD | TBD | TBD |
| verify 32 | router | 256 | 256 | 8192 | TBD | TBD | TBD |

Use `benchmarks/compare.py` on a complete measured result to generate this
comparison with real numbers and the six weighted load totals. Partial runs
must not be scored. The H200 marketing bandwidth of 4.8 TB/s and BF16 peak
989 TFLOP/s in the task describe idealized lower bounds, never measured results.
