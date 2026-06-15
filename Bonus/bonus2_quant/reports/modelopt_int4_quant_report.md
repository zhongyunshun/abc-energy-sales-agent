# Bonus 2 ModelOpt Quantization Run

- format: `int4_awq`
- recipe: `mtq.INT4_AWQ_CFG`
- model: `/workspace/models/adapters/merged`
- artifact: `/workspace/Bonus/bonus2_quant/artifacts/modelopt_int4_awq`
- calibration: 32 samples, max_seq_len 512, per-scenario {'closing': 2, 'cold_open': 2, 'general': 19, 'info_gathering': 4, 'objection_handling': 5}

## Size

| Model | GB | Bytes |
|---|---:|---:|
| FP16 merged | 8.045 | 8,044,981,992 |
| ModelOpt int4_awq | 2.71 | 2,709,581,448 |

- compression ratio: 2.969x
- size reduction: 66.32%

## Fake-Quant Probe

### quant-probe-01-objection

**FP16:** I understand your concern about pricing. While our rates are competitive, the real value comes from our commitment to renewable energy and transparent billing. We aim to provide long-term savings and environmental benefits.

**ModelOpt:** I understand your concern. While our rates are competitive, the real value comes from our comprehensive service and how we tailor plans to your specific usage. We often find that once we analyze your actual energy consumption, the long-term savings and benefits outweigh the initial comparison.

### quant-probe-02-info

**FP16:** <think>

</think>

Great! To help you find the best electricity plan, could you tell me your average monthly electricity usage or bill amount?

**ModelOpt:** <think>

</think>

Great! To help you find the best electricity plan, could you tell me your average monthly electricity usage or the amount you typically pay? This will allow us to compare options and find the most suitable one for you.

### quant-probe-03-pricing

**FP16:** I can't give you an exact rate without knowing your specific usage, but I can tell you that our fixed plans offer predictable pricing. To give you the most accurate details, I'd need to look at your historical usage.

**ModelOpt:** I can't give you an exact rate without knowing your specific usage. However, I can tell you that our fixed plans offer predictable pricing, and I can look at your past usage to give you a personalized quote.

### quant-probe-04-coldopen

**FP16:** <think>

</think>

Hello! My name is Sarah, and I am calling from TechSolutions. We are reaching out to local businesses like yours to discuss potential partnerships and how we can help streamline your operations.

**ModelOpt:** <think>

</think>

Hello! My name is Sarah, and I work for a tech company. We are reaching out to local businesses to offer a free consultation on their digital marketing strategies. I understand you might be busy, but I believe this could be a great opportunity for your business.

### quant-probe-05-closing

**FP16:** Great! The next step is a quick verbal verification to confirm your details. After that, we'll send you a welcome pack with all the terms in writing for your review.

**ModelOpt:** Great! The next step is a quick verbal verification to confirm your details. After that, we'll send you a welcome pack with all the terms in writing for your review.
