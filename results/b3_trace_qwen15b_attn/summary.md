# B3 decode trace, Qwen/Qwen2.5-1.5B (git 0314dae4c2696b210c0a3c123c03f49527ba2166)

Prompt 64 tokens, 16 cached decode steps, config `b3_trace_fq_n128_attn.yaml`, attention on CIM: True. 17 of 17 checks equal (measured event counts = counts derived from the model shape).

| Check | MACs measured | MACs derived | ADC conversions measured | derived | Equal |
|---|---|---|---|---|---|
| prefill, 64 tokens | 84031438848 | 84031438848 | 42147151872 | 42147151872 | True |
| decode step 1 (position 65) | 1315786752 | 1315786752 | 659248128 | 659248128 | True |
| decode step 16 (position 80) | 1317076992 | 1317076992 | 659570688 | 659570688 | True |
