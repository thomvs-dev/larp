# A2 — Reproduction of ASiM on ResNet-20 / CIFAR-10

Status: full sweep done (1,000 images, 13 configs). GATE A2 conditions 1–3 pass; condition 4 (ADC ≤ 6 b within run-to-run spread) waits for the spread run (section "Spread run").

## Setup

- Model: CIFAR-10 ResNet-20, pretrained weights from chenyaofo/pytorch-cifar-models (hash-checked), no retraining.
- Both simulators run the same weights. All conv layers except the stem run on the simulated macro; stem conv, BatchNorm, ReLU, pooling and classifier stay in float.
- Quantization: W8A8, per tensor, unsigned activations (ASiM conv convention). Same batches (100 images) for both, since ASiM's activation scale is the batch max.
- Images: first 1,000 CIFAR-10 test images (CPU budget; `docs/cpu_guide.md`).

## What is already established (before the sweep)

- Layer level: cimsim's `CIMConv2d` matches `ASiMConv2d` to ≤ 1e-6 relative for zero noise, bit-serial and bit-parallel, 1×1 and 3×3, stride 1 and 2, signed and unsigned (`tests/test_calib_asym_conv.py`).
- Inside the network: with ASiM's inputs fed to each of our conv layers, all 20 layers match to 1.3e-6 (nrow 128, ADC 6 b).
- Known difference: ASiM skips channel groups for some nrow values (findings F4). Config `bs_n64_adc8` shows it.
- Expected spread: at low ADC precision, whole-network predictions are sensitive to ~1e-7 float differences (findings F6), so per-config accuracy differences of a few % at ADC ≤ 6 b are not evidence of a semantic difference.

## GATE A2 criterion (refined, approved by Thomas 2026-11-02)

- Trends (accuracy vs ADC bits, nrow, noise, encoding) match.
- Per-config difference ≤ 1% absolute at ADC ≥ 8 b and for fake-quant; at ADC ≤ 6 b, difference within the run-to-run spread of either simulator.
- `bs_n64_adc8` difference explained by F4.

## Results

### Smoke run (200 images) — `results/a2_smoke/`

| Config | cimsim acc (%) | ASiM acc (%) | diff | cimsim s | ASiM s |
|---|---|---|---|---|---|
| ideal_w8a8 (fake-quant) | 92.00 | 92.00 | +0.00 | 0.5 | 0.4 |
| bs_n128_adc8 | 92.00 | 92.00 | +0.00 | 11.2 | 102.3 |
| bs_n128_adc6 | 59.00 | 60.00 | -1.00 | 12.4 | 103.3 |

FP32 92.50% on these 200 images. cimsim is ~9× faster than ASiM. The ADC 6 b drop (92 → 59–60%) appears in both simulators.

### Full sweep (1,000 images) — `results/a2_resnet20/`

Git `bcde8c5b`, i7-12700H, 14 threads, batch 100, cimsim backend `fast`. FP32 on these images: 92.50%.

| Config | What changes | cimsim (%) | ASiM (%) | diff | cimsim s | ASiM s | speed-up |
|---|---|---|---|---|---|---|---|
| ideal_w8a8 | fake-quant W8A8 | 92.60 | 92.60 | +0.00 | 5.0 | 3.7 | 0.7× |
| bs_n128_adc8 | reference point | 92.40 | 92.40 | +0.00 | 132.4 | 892.4 | 6.7× |
| bs_n128_adc6 | ADC bits | 60.20 | 59.00 | +1.20 | 133.7 | 697.4 | 5.2× |
| bs_n128_adc5 | ADC bits | 12.80 | 13.10 | −0.30 | 58.7 | 505.4 | 8.6× |
| bs_n128_adc4 | ADC bits | 9.90 | 10.30 | −0.40 | 58.0 | 687.6 | 11.9× |
| bs_n255_adc8 | nrow | 92.60 | 92.60 | +0.00 | 6.9 | 457.0 | 66× |
| bs_n255_adc6 | nrow | 13.00 | 13.00 | +0.00 | 95.9 | 542.9 | 5.7× |
| bs_n64_adc8 | nrow (F4) | 92.40 | 90.30 | +2.10 | 183.1 | 1406.6 | 7.7× |
| bs_n128_adc8_rn1 | random noise 1% | 10.40 | 9.60 | +0.80 | 350.2 | 609.3 | 1.7× |
| bs_n128_adc8_rn2 | random noise 2% | 9.20 | 10.00 | −0.80 | 345.7 | 612.0 | 1.8× |
| bs_n128_adc8_nl2 | nonlinearity 2% | 9.30 | 10.00 | −0.70 | 381.2 | 617.6 | 1.6× |
| bp2_n128_adc8 | 2-bit input cycles | 85.70 | 85.40 | +0.30 | 108.2 | 505.6 | 4.7× |
| bp2_n128_adc6 | 2-bit input cycles | 12.90 | 13.10 | −0.20 | 97.6 | 482.6 | 4.9× |

#### Gate evaluation

| Condition | Configs | Result |
|---|---|---|
| 1. Trends match (ADC bits, nrow, noise, encoding) | all | **Pass.** Every sweep direction moves the same way by about the same amount in both simulators (below). |
| 2. Fake-quant and ADC ≥ 8 b: \|diff\| ≤ 1% | ideal, n128/n255 adc8, rn1, rn2, nl2, bp2_adc8 | **Pass.** Max \|diff\| 0.8% (noise configs, both at chance level). |
| 3. `bs_n64_adc8` explained by F4 | bs_n64_adc8 | **Pass.** cimsim gives 92.40%, identical to nrow 128 at the same ADC, as expected when every channel is computed. ASiM gives 90.30%: with nrow 64 and 3×3 kernels it splits 64-channel inputs into 10 channel groups but computes 9, so one input channel is missing in every conv with 64 input channels (F4). The 2.1% is the cost of that dropped channel, not a semantic difference. |
| 4. ADC ≤ 6 b: \|diff\| within run-to-run spread | n128 adc6/5/4, n255_adc6, bp2_adc6 | **Pending.** Diffs are −0.4 … +1.2%. F6 predicts spread of this size, but the criterion needs it measured — see "Spread run". |

#### What the sweep shows (both simulators)

- **ADC bits (nrow 128, bit-serial):** 92.4 → 60 → 13 → 10% for 8 → 6 → 5 → 4 b. With ASiM's full-scale ADC range (full scale = nrow MAC units), the ADC step is 128/255 ≈ 0.5 MAC unit at 8 b (lossless), 128/63 ≈ 2 units at 6 b and ≈ 4 units at 5 b. Accuracy drops as soon as the step exceeds one MAC unit — the same mechanism as F3/F5 on LLMs.
- **nrow:** nrow 255 with ADC 8 b gives 255 levels for 255 MAC units, a step of exactly one unit, so the readout is exact (92.6% = fake-quant). This is also why cimsim needs 6.9 s for it: the fast backend detects the lossless case and skips the ADC model. nrow 255 with ADC 6 b (step ≈ 4 units) collapses to 13%, worse than nrow 128 at 6 b. More rows per conversion need more ADC bits at a fixed range.
- **Noise:** 1% or 2% Gaussian noise (relative to full scale), or 2% nonlinearity, at ADC 8 b takes the network to chance (~10%) in both. 1% of a 128-unit full scale is 1.3 MAC units of noise per conversion, multiplied by shift-add weights up to 2^14. ResNet-20 with per-tensor W8A8 has no tolerance for this without noise-aware training.
- **Bit-parallel inputs (2 bits per cycle):** full scale becomes 3 × 128 = 384 units over 255 levels at 8 b (step 1.5 units): 85.7 / 85.4%, 7% below bit-serial. At 6 b both collapse.
- **Speed:** cimsim is 1.6–66× faster than ASiM on every bit-accurate config (≈ 5–12× noise-free, ≈ 1.7× with noise, 66× when lossless). ASiM is faster only for fake-quant (no array model).

### Spread run (for condition 4)

cimsim's `fast` and `reference` backends have the same semantics (tested ≤ 1e-5 noise-free) and differ only in float rounding. Running the five low-ADC configs with both backends and comparing image by image measures the run-to-run spread of one simulator caused by float rounding alone — the cause F6 identifies for the cimsim–ASiM gap. The fast run also re-runs ASiM so cimsim–ASiM disagreement can be counted per image.

```
python validation/a2_asim_compare.py --name a2_spread_ref  --backend reference --sim cimsim --configs bs_n128_adc6,bs_n128_adc5,bs_n128_adc4,bs_n255_adc6,bp2_n128_adc6
python validation/a2_asim_compare.py --name a2_spread_fast --backend fast      --sim both   --configs bs_n128_adc6,bs_n128_adc5,bs_n128_adc4,bs_n255_adc6,bp2_n128_adc6
python validation/a2_spread.py results/a2_spread_fast results/a2_spread_ref
```

Decision rule: condition 4 passes if, for each config, \|cimsim − ASiM\| is no larger than the largest \|fast − reference\| accuracy difference over the five configs, or if both pairs disagree on a similar fraction of images. If fast and reference agree exactly while cimsim and ASiM do not, the gap is not explained by float rounding and needs a layer-by-layer check at those configs.

(results to be filled)
