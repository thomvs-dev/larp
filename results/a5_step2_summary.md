# A5 step 2 — statistical mode at 8 transition points

Thomas's laptop, git b5f6e16, 14 threads, first 8,192 WikiText-2 test tokens, noise-free, other layers at calibrated ADC 8 b (`a5_base_adc8cal.yaml`). Bit-accurate reference: step 1 rerun (git 695e2ba). Pass = statistical within 2 % of bit-accurate. Full scales are identical in both modes at every point (`adc_full_scale.json`).

| Model | Point (role:nrow:bits) | Bit-accurate | Statistical | Stat / BA | Full scale of the swept layers | ADC step (MAC units) | 2 % rule |
|---|---|---|---|---|---|---|---|
| OPT-125M | qkv:64:5 | 31.72 | 31.724 | 1.000 | 31 | 1 | pass |
| OPT-125M | qkv:128:6 | 30.97 | 30.971 | 1.000 | 63 | 1 | pass |
| OPT-125M | all:64:5 | 31.94 | 31.936 | 1.000 | 31 | 1 | pass |
| OPT-125M | o:256:6 | 43.18 | 324.64 | 7.5 | 81–83 (12 of 12 layers) | 1.29–1.32 | fail |
| GPT-2 | up_gate:64:5 | 34.23 | 34.230 | 1.000 | 31 | 1 | pass |
| GPT-2 | o:64:5 | 33.82 | 33.822 | 1.000 | 31 | 1 | pass |
| GPT-2 | down:256:7 | 69.15 | 214,821 | 3,107 | 127 (7 layers), 132–152 (5 layers) | 1.04–1.20 on 5 layers | fail |
| GPT-2 | all:256:7 | 68.80 | 212,641 | 3,091 | as down:256:7 for `mlp.c_proj` | 1.04–1.20 on 5 layers | fail |

Sources: `results/a5s2_opt125m_ba/`, `results/a5s2_gpt2_ba/`, `results/a5v2_opt125m_stat/`, `results/a5v2_gpt2_stat/`.

## Reading

- 5 of 8 points pass, each within 0.02 %. At all five every swept layer has an ADC step of exactly 1 MAC unit, so the only error is saturation, which statistical mode computes exactly.
- The 3 failing points are the ones where MSE calibration chose a full scale above 2^b − 1 (less clipping, step above 1 MAC unit and not 1/n). Rounding error is then non-zero on most conversions and correlated across bit planes; statistical mode gives it to the analytic model, which assumes independent conversions and overestimates it (findings F13).
- Layer-level confirmation on a synthetic layer (`results/a5_step_check_sandbox/result.json`): statistical = bit-accurate at steps 1 and 1/3; at steps 1.04–2.0 the statistical error is 1.4–4.3× the bit-accurate error.

## Consequence (decision 2026-11-13, CLAUDE.md §8)

Statistical mode is used only at points where every statistical layer has an ADC step of 1/n MAC unit. `run_ppl.py` records `stat_inexact_layers` in `result.json` and prints a warning; `a5_sweep.py` marks such points `*` in `summary.md`. Points in the gradual-degradation regime (OPT o-proj, GPT-2 `mlp.c_proj` near their boundary) are reported from bit-accurate runs on OPT-125M / GPT-2 only.

## Side note: fake-quant reference 30.449 vs 30.460 (OPT-125M)

`a5s2_opt125m_ba/fake_quant` ran with 7 threads (30.449); `a5v2_opt125m_stat/fake_quant` with 14 threads (30.460), which equals the lossless bit-accurate points (14 threads). The commits in between changed no code. The earlier "+0.04 % offset of lossless bit-accurate vs fake-quant" is therefore most likely the thread count of the reference run, not the executor. Not yet confirmed by a 7-thread rerun.
