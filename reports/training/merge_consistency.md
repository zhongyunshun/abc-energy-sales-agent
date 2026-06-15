# M6 merge consistency check — 8 prompts (greedy)

**PASS** — 8/8 prompts match (mode=exact).

Each fixed prompt is greedy-generated on `base + DPO adapter` (PEFT inference) and on the merged dense model; the continuations are compared. A PASS means the merge is behaviour-preserving.

| base: unsloth/Qwen3-4B-Instruct-2507 | adapter: /u1/x2jian/neo/sales_agent/models/adapters/dpo | dtype: bfloat16 | device_map: auto | merged_size_gb: 8.045 | git_commit: 78033631f916e610aa7ed349d01e50ed34f29d1e | created_at: 2026-06-14T01:37:24.148256+00:00 |


## merge-consistency-01-objection — PASS (exact)

**Output (identical on base+adapter and merged):**

I understand your concern about pricing. While our rates are competitive, the real value comes from our commitment to renewable energy and transparent billing. We aim to provide long-term savings and environmental benefits.


## merge-consistency-02-info — PASS (exact)

**Output (identical on base+adapter and merged):**

<think>

</think>

Great! To help you find the best electricity plan, could you tell me your average monthly electricity usage or bill amount?


## merge-consistency-03-pricing — PASS (exact)

**Output (identical on base+adapter and merged):**

I can't give you an exact rate without knowing your specific usage, but I can tell you that our fixed plans offer predictable pricing. To give you the most accurate details, I'd need to look at your historical usage.


## merge-consistency-04-coldopen — PASS (exact)

**Output (identical on base+adapter and merged):**

<think>

</think>

Hello! My name is Sarah, and I am calling from TechSolutions. We are reaching out to local businesses like yours to discuss potential partnerships and how we can help streamline your operations.


## merge-consistency-05-closing — PASS (exact)

**Output (identical on base+adapter and merged):**

Great! The next step is a quick verbal verification to confirm your details. After that, we'll send you a welcome pack with all the terms in writing for your review.


## merge-consistency-06-objection — PASS (exact)

**Output (identical on base+adapter and merged):**

<think>

</think>

I understand that you have a commitment with your current supplier. However, I can still provide you with information about our products and services, so you can make an informed decision when your contract is up for renewal.


## merge-consistency-07-info — PASS (exact)

**Output (identical on base+adapter and merged):**

That's great to hear you're using renewable energy! We have plans designed to complement various energy setups. To find the best fit, could you tell me a bit about your average monthly electricity usage or bill amount?


## merge-consistency-08-multiturn — PASS (exact)

**Output (identical on base+adapter and merged):**

That gives us a good starting point. To see how we can best help, do you know when your current electricity contract is set to expire?
