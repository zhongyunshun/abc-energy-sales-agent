# Bonus 2 ModelOpt Quantization Run

- format: `fp8`
- recipe: `mtq.FP8_DEFAULT_CFG`
- model: `/workspace/models/adapters/merged`
- artifact: `/workspace/Bonus/bonus2_quant/artifacts/modelopt_fp8`
- calibration: 64 samples, max_seq_len 768, per-scenario {'closing': 3, 'cold_open': 3, 'general': 39, 'info_gathering': 8, 'objection_handling': 11}

## Size

| Model | GB | Bytes |
|---|---:|---:|
| FP16 merged | 8.045 | 8,044,981,992 |
| ModelOpt fp8 | 4.412 | 4,411,718,392 |

- compression ratio: 1.824x
- size reduction: 45.16%

## Fake-Quant Probe

### quant-probe-01-objection

**FP16:** I understand your concern about pricing. While our rates are competitive, the real value comes from our commitment to renewable energy and transparent billing. We aim to provide long-term savings and environmental benefits.

**ModelOpt:** I understand your concern about pricing. While our rates are competitive, the real value comes from our commitment to transparent billing and excellent customer service. We aim to provide long-term savings and peace of mind.

### quant-probe-02-info

**FP16:** <think>

</think>

Great! To help you find the best electricity plan, could you tell me your average monthly electricity usage or bill amount?

**ModelOpt:** <think>

</think>

Great! To help me find the best plan for you, could you tell me your average monthly electricity usage or bill amount?

### quant-probe-03-pricing

**FP16:** I can't give you an exact rate without knowing your specific usage, but I can tell you that our fixed plans offer predictable pricing. To give you the most accurate details, I'd need to look at your historical usage.

**ModelOpt:** I can provide you with a personalized quote based on your specific usage. To do that, I'll need to look at your historical energy consumption.

### quant-probe-04-coldopen

**FP16:** <think>

</think>

Hello! My name is Sarah, and I am calling from TechSolutions. We are reaching out to local businesses like yours to discuss potential partnerships and how we can help streamline your operations.

**ModelOpt:** <think>

</think>

Hello! My name is Sarah, and I am calling from TechSolutions. We are reaching out to local businesses like yours to discuss potential partnerships and how we can help streamline your operations.

### quant-probe-05-closing

**FP16:** Great! The next step is a quick verbal verification to confirm your details. After that, we'll send you a welcome pack with all the terms in writing for your review.

**ModelOpt:** Great! The next step is a quick verbal verification to confirm your details. After that, we'll send you a welcome pack with all the terms in writing for your review.
