# A2 — Reproduction of ASiM on ResNet-20 / CIFAR-10

Status: smoke run done (200 images, 3 configs, all within 1%); full sweep (1,000 images, 13 configs) pending.

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

## GATE A2 criterion (refined)

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

### Full sweep (1,000 images)

(fill from `results/a2_resnet20/table.md`)
