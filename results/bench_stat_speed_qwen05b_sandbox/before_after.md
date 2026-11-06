# Statistical-mode speed, Qwen2.5-0.5B, cloud sandbox (2 threads)

Model: Qwen2.5-0.5B (local copy), config `config/experiments/a4_stat_adc6cal.yaml` (W8A8, bit-serial,
nrow 128, ADC 6 b calibrated, noise-free), calibration 1,024 WikiText-2 train tokens. All times are
ms per token for one forward call of the given shape (batch x tokens); 8 x 110 is the shape of a
zero-shot batch (batch 8, ~110 tokens per request), 1 x 2048 a perplexity window.

`result.json` (this folder): code at fa06edf, model calibrated once and loaded (`--load`).

| Shape | before (c5b4383) | after (fa06edf) | fake-quant floor | after / floor |
|---|---|---|---|---|
| 1 x 64 | 42.7 | 28.5 | 11.1 | 2.6 |
| 8 x 110 | 22.7 | 15.6 | 5.4 | 2.9 |
| 1 x 2048 | 22.2 | 15.5 | 5.8 | 2.7 |

"before" was measured on the same calibrated model and machine with the code at c5b4383
(scratch timing script, 3 repeats for ≤ 256 tokens, 1 for longer calls); "after" is from `result.json`.
Earlier history for 1 x 64: 85.8 (loop path only, 2ce11a2) -> 49.1 (batched path, c5b4383) -> 28.5.

Output identity: the logits of fa06edf and c5b4383 are bit-identical (`torch.equal`) for shapes
1 x 64, 1 x 128, 1 x 256, 8 x 110 and 1 x 2048; loop and batched paths give identical logits at
1 x 64 (`paths_identical` in `result.json`). 254 tests pass.

Changes in fa06edf:
- exact saturation term (both paths): conversions with m below the smallest MAC value that has a
  non-zero table entry are skipped before the int64 conversion and table gather; the loop path
  computes the digit slice and row sums once per (chunk, cycle) instead of once per entry;
- statistical matmul in float32 when the Cauchy-Schwarz bound max_t ||qx_t|| * max_n ||qw_n|| < 2^23
  (every partial sum is then an integer below 2^24, so float32 is exact), else float64.

Open: Thomas's laptop measured 51–73 ms/token for statistical zero-shot with the old code (14 threads)
while the sandbox gives 22.7 on 2 threads for the same shape; the laptop W8A8 run was also slower
than the sandbox fake-quant floor (7.6–8.4 vs 5.4). Thread count is the first suspect (many small ops
per call); `bench_real_short_calls.py --threads 14,8,6,4` on the laptop measures it.
