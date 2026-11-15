# Part B cost report, Qwen/Qwen2.5-1.5B (git ff767af)

Run `a4_qwen15b_stat_adc6cal` (git 2d1d11f, PPL 9.492), nrow 128, ADC 6 b, 22 nm / 0.8 V, DRAM `gddr5` 14.00 pJ/bit.
Count check: derived totals equal `events.json` (299008 tokens processed).
One conversion: array 164 fJ, ADC 357 fJ, shift-add + register 46 fJ. Array write 90 fJ/bit.

Energy per token, mJ. Columns: array + ADC + shift-add (compute), DRAM read of streamed weights, array write of streamed weights, attention + KV cache, non-linear units, LM head (digital MACs + its weights from DRAM).

| Budget (Mbit) | Resident | Cell area (mm²) | KV | Phase | Seq | Total | Compute | DRAM weights | Array write | Attn + KV | Units | LM head | Tokens/s bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 110 | 0 % | 23 | buffer | prefill | 512 | 0.716 | 0.372 | 0.287 | 0.002 | 0.0041 | 0.0007 | 0.051 | 4152.2 |
| 110 | 0 % | 23 | buffer | decode | 512 | 174.231 | 0.372 | 146.742 | 0.946 | 0.0081 | 0.0009 | 26.162 | 8.3 |
| 512 | 4 % | 105 | buffer | prefill | 512 | 0.705 | 0.372 | 0.276 | 0.002 | 0.0041 | 0.0007 | 0.051 | 4287.9 |
| 512 | 4 % | 105 | buffer | decode | 512 | 168.602 | 0.372 | 141.149 | 0.910 | 0.0081 | 0.0009 | 26.162 | 8.6 |
| 2048 | 18 % | 422 | buffer | prefill | 512 | 0.663 | 0.372 | 0.234 | 0.002 | 0.0041 | 0.0007 | 0.051 | 4904.9 |
| 2048 | 18 % | 422 | buffer | decode | 512 | 146.927 | 0.372 | 119.613 | 0.771 | 0.0081 | 0.0009 | 26.162 | 9.8 |
| 10482 | 100 % | 2161 | buffer | prefill | 512 | 0.428 | 0.372 | 0.000 | 0.000 | 0.0041 | 0.0007 | 0.051 | 24439.7 |
| 10482 | 100 % | 2161 | buffer | decode | 512 | 26.543 | 0.372 | 0.000 | 0.000 | 0.0081 | 0.0009 | 26.162 | 54.8 |

Limits: cell area only (no ADC / periphery area in the M1 anchor); latency is a bound from DRAM bandwidth plus one array evaluation per input cycle and layer (digital units and scheduling not timed); analog costs are the M1 macro moved from 65 nm to the result node by the scaling rule (docs/assumptions.md B-11, B-25).
