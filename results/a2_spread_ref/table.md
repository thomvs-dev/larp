# A2 / E1 - resnet20 (chenyaofo, CIFAR-10), 1000 CIFAR-10 test images

FP32: 92.50%. Git 9c6486c7. Stem conv and classifier in float.

| Config | cimsim acc (%) | ASiM acc (%) | diff (cimsim - ASiM) | cimsim s | ASiM s |
|---|---|---|---|---|---|
| bs_n128_adc6 | 61.30 |  |  | 139.3 |  |
| bs_n128_adc5 | 13.30 |  |  | 150.2 |  |
| bs_n128_adc4 | 9.50 |  |  | 145.1 |  |
| bs_n255_adc6 | 11.60 |  |  | 109.7 |  |
| bp2_n128_adc6 | 11.60 |  |  | 117.0 |  |
