# A2 / E1 - resnet20 (chenyaofo, CIFAR-10), 1000 CIFAR-10 test images

FP32: 92.50%. Git 9c6486c7. Stem conv and classifier in float.

| Config | cimsim acc (%) | ASiM acc (%) | diff (cimsim - ASiM) | cimsim s | ASiM s |
|---|---|---|---|---|---|
| bs_n128_adc6 | 60.20 | 59.00 | +1.20 | 63.3 | 509.0 |
| bs_n128_adc5 | 12.80 | 13.10 | -0.30 | 61.8 | 511.8 |
| bs_n128_adc4 | 9.90 | 10.30 | -0.40 | 61.6 | 545.4 |
| bs_n255_adc6 | 13.00 | 13.00 | +0.00 | 52.6 | 368.1 |
| bp2_n128_adc6 | 12.90 | 13.10 | -0.20 | 48.8 | 336.1 |
