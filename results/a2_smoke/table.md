# A2 / E1 - resnet20 (chenyaofo, CIFAR-10), 200 CIFAR-10 test images

FP32: 92.50%. Git 174d32f0. Stem conv and classifier in float.

| Config | cimsim acc (%) | ASiM acc (%) | diff (cimsim - ASiM) | cimsim s | ASiM s |
|---|---|---|---|---|---|
| ideal_w8a8 | 92.00 | 92.00 | +0.00 | 0.5 | 0.4 |
| bs_n128_adc8 | 92.00 | 92.00 | +0.00 | 11.2 | 102.3 |
| bs_n128_adc6 | 59.00 | 60.00 | -1.00 | 12.4 | 103.3 |
