# M7 AWQ quantization report

- git_commit: `360cdd376d5d42b999097c32aa7a33cf6776d79b`
- created_at: 2026-06-14T04:46:44.306747+00:00
- merged (FP16) dir: `/workspace/models/adapters/merged`
- recipe: W4A16_ASYM (group_size 128, symmetric=False, duo_scaling=both)
- calibration: 256 samples, max_seq_len 768, per-scenario {'closing': 11, 'cold_open': 12, 'general': 155, 'info_gathering': 33, 'objection_handling': 45}
- device_map: cpu, pipeline: sequential

## Size trade-off (README source)

| | bytes | GB |
|---|---:|---:|
| FP16 (merged) | 8,044,981,992 | 8.045 |
| INT4 (AWQ) | 2,666,063,024 | 2.666 |

- compression ratio: **3.018x**
- size reduction: **66.86%**
- self-check generation: PASS (`<think>

</think>

Of course! I can assist you in finding ways to reduce your electricity costs. Could you tell me more about your current usage and any specific`)


# M7 AWQ FP16-vs-INT4 probe (greedy)

## quant-probe-01-objection

**FP16 (merged):** I understand your concern about pricing. While our rates are competitive, the real value comes from our commitment to renewable energy and transparent billing. We aim to provide long-term savings and environmental benefits.

**INT4 (AWQ):** I understand that price is a key factor. While our rates might appear different at first glance, we focus on providing value through our service and plan flexibility. We can analyze your current usage to see if our plans offer better long-term benefits.

## quant-probe-02-info

**FP16 (merged):** <think>

</think>

Great! To help you find the best electricity plan, could you tell me your average monthly electricity usage or bill amount?

**INT4 (AWQ):** <think>

</think>

Great! To help you find the best electricity plan, could you tell me your average monthly electricity usage or bill amount? This will give us a better idea of your energy consumption.

## quant-probe-03-pricing

**FP16 (merged):** I can't give you an exact rate without knowing your specific usage, but I can tell you that our fixed plans offer predictable pricing. To give you the most accurate details, I'd need to look at your historical usage.

**INT4 (AWQ):** I can't give you an exact rate without knowing your specific usage. However, I can tell you that our fixed plans are designed to offer stability, and we can review your historical usage to provide a personalized quote.

## quant-probe-04-coldopen

**FP16 (merged):** <think>

</think>

Hello! My name is Sarah, and I am calling from TechSolutions. We are reaching out to local businesses like yours to discuss potential partnerships and how we can help streamline your operations.

**INT4 (AWQ):** <think>

</think>

Hello! My name is Sarah, and I am calling from TechSolutions. We are reaching out to local businesses like yours to offer our innovative software solutions. I understand you might be busy, but I wanted to briefly explain how our services can benefit your business.

## quant-probe-05-closing

**FP16 (merged):** Great! The next step is a quick verbal verification to confirm your details. After that, we'll send you a welcome pack with all the terms in writing for your review.

**INT4 (AWQ):** Great! The next step is a quick verbal verification to confirm your details. After that, we'll send you a welcome pack with all the terms in writing for your review.
