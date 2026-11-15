# B3 decode trace, Qwen/Qwen2.5-1.5B (git 0314dae4c2696b210c0a3c123c03f49527ba2166)

Prompt 64 tokens, 16 cached decode steps, config `b3_trace_fq_n128.yaml`, attention on CIM: False. 17 of 17 checks equal (measured event counts = counts derived from the model shape).

| Check | MACs measured | MACs derived | ADC conversions measured | derived | Equal |
|---|---|---|---|---|---|
| prefill, 64 tokens | 83852525568 | 83852525568 | 41926262784 | 41926262784 | True |
| decode step 1 (position 65) | 1310195712 | 1310195712 | 655097856 | 655097856 | True |
| decode step 16 (position 80) | 1310195712 | 1310195712 | 655097856 | 655097856 | True |
