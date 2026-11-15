# Figure A6 data

| Model | Point | Variant | PPL | / W8A8 | analog conversions / token | DCIM ops / token | Hadamard adds / token |
|---|---|---|---|---|---|---|---|
| GPT-2 | n128 / adc6 | no mitigation | 198.377 | 5.873 | 4.25e+07 | 0 | 0 |
| GPT-2 | n128 / adc6 | Hadamard, down input | 33.970 | 1.006 | 4.25e+07 | 0 | 3.69e+05 |
| GPT-2 | n128 / adc6 | Hadamard, all inputs | 33.583 | 0.994 | 4.25e+07 | 0 | 5.9e+05 |
| GPT-2 | n128 / adc6 | down on DCIM | 34.013 | 1.007 | 2.83e+07 | 1.42e+07 | 0 |
| GPT-2 | n128 / adc6 | down +1 ADC bit | 34.013 | 1.007 | 4.25e+07 | 0 | 0 |
| GPT-2 | n256 / adc7 | no mitigation | 68.797 | 2.037 | 2.12e+07 | 0 | 0 |
| GPT-2 | n256 / adc7 | Hadamard, down input | 33.741 | 0.999 | 2.12e+07 | 0 | 3.69e+05 |
| GPT-2 | n256 / adc7 | Hadamard, all inputs | 33.722 | 0.998 | 2.12e+07 | 0 | 5.9e+05 |
| GPT-2 | n256 / adc7 | down on DCIM | 33.877 | 1.003 | 1.42e+07 | 7.08e+06 | 0 |
| GPT-2 | n256 / adc7 | down +1 ADC bit | 33.877 | 1.003 | 2.12e+07 | 0 | 0 |
| OPT-125M | n128 / adc6 | no mitigation | 30.816 | 1.012 | 4.25e+07 | 0 | 0 |
| OPT-125M | n128 / adc6 | Hadamard, all inputs | 30.104 | 0.988 | 4.25e+07 | 0 | 5.9e+05 |
| Qwen2.5-1.5B | n128 / adc6 | no mitigation | 10.342 | 1.000 | 6.55e+08 | 0 | 0 |
| Qwen2.5-1.5B | n128 / adc6 | Hadamard, all inputs | 10.229 | 0.989 | 6.55e+08 | 0 | 3.17e+06 |
