# Bonus 2: NVIDIA ModelOpt FP8 Quantization

This folder contains an advanced quantization path using NVIDIA Model Optimizer
(ModelOpt) as a comparison to the main M7 llm-compressor AWQ path.

Large checkpoints are written under `Bonus/bonus2_quant/artifacts/` and are
ignored by git.

## Result Summary

The FP8 path ran successfully on the PC RTX 4070 (Ada, sm89) and produced a
native vLLM-loadable ModelOpt checkpoint:

- input model: `models/adapters/merged`
- output checkpoint: `Bonus/bonus2_quant/artifacts/modelopt_fp8`
- ModelOpt version: `0.44.0`
- actual config used: `mtq.FP8_DEFAULT_CFG`
- exported quant config: `quant_algo=FP8`, `kv_cache_quant_algo=null`,
  `exclude_modules=["lm_head"]`
- calibration: 64 stratified training-domain samples, `max_seq_len=768`
- size: FP16 `8.045 GB` -> ModelOpt FP8 `4.412 GB` (`1.824x`, `45.16%` smaller)
- native vLLM smoke: PASS, 5/5 probe generations non-empty

This FP8 result is a hardware-path comparison, not an apples-to-apples size
comparison against the M7 INT4 AWQ artifact.

To make the size comparison fair, a supplemental ModelOpt INT4_AWQ run was also
executed:

- output checkpoint: `Bonus/bonus2_quant/artifacts/modelopt_int4_awq`
- actual config used: `mtq.INT4_AWQ_CFG`
- exported quant config: `quant_algo=W4A16_AWQ`, `group_size=128`,
  `has_zero_point=false`, `pre_quant_scale=true`, `exclude_modules=["lm_head"]`
- calibration: 32 stratified training-domain samples, `max_seq_len=512`
- size: FP16 `8.045 GB` -> ModelOpt INT4_AWQ `2.710 GB` (`2.969x`,
  `66.32%` smaller)
- fake-quant probe: 5/5 generated, but one cold-open response drifted off the
  energy-sales domain
- native vLLM smoke: FAIL, because vLLM v0.10.2 supports ModelOpt `FP8` and
  `NVFP4`, not ModelOpt `W4A16_AWQ`

The originally planned recipe name,
`general/ptq/fp8_default-kv_fp8_cast`, is documented by the current ModelOpt docs
but was not present as a loadable recipe in the `nvidia-modelopt==0.44.0` wheel
used here. The script records the fallback and uses the stable release constant
`mtq.FP8_DEFAULT_CFG`.

## Files

| Path | Purpose |
|---|---|
| `quantize_modelopt.py` | Container-only ModelOpt quantization script. |
| `smoke_vllm_modelopt.py` | Self-contained native vLLM smoke script. |
| `reports/modelopt_manifest.json` | Run manifest, calibration mix, sizes, output path. |
| `reports/modelopt_quant_report.md` | FP16 vs ModelOpt fake-quant probe comparison. |
| `reports/fp8_vllm_native_smoke.jsonl` | Native vLLM outputs for the same 5 M7 probes. |
| `reports/modelopt_int4_manifest.json` | Supplemental same-bit-width INT4_AWQ manifest. |
| `reports/modelopt_int4_quant_report.md` | INT4_AWQ FP16-vs-fake-quant probe comparison. |
| `reports/modelopt_int4_vllm_native_smoke.md` | vLLM unsupported-format evidence. |
| `artifacts/modelopt_fp8/` | Large FP8 checkpoint, gitignored. |
| `artifacts/modelopt_int4_awq/` | Large INT4_AWQ checkpoint, gitignored. |

## Install And Run

All GPU work was run inside Docker, not on the Windows host. The train image did
not include ModelOpt, so the script installs it inside the temporary container.

The first attempt with `nvidia-modelopt[all]` was too broad for this image: it
pulled a different torch/transformers stack and hit a torchvision operator
mismatch. The final script uses a conservative install:

- `nvidia-modelopt==0.44.0 --no-deps`
- minimal base deps only: `ninja`, `omegaconf`, `pulp`, `rich`, `setuptools`

Example command:

```bash
docker compose -f docker/compose.yaml run --rm \
  --volume <repo>/Bonus:/workspace/Bonus \
  --entrypoint uv train run python Bonus/bonus2_quant/quantize_modelopt.py \
  --install-missing-modelopt \
  --modelopt-install-spec nvidia-modelopt==0.44.0 \
  --format fp8 \
  --recipe general/ptq/fp8_default-kv_fp8_cast \
  --n-calib 64 \
  --max-seq-len 768 \
  --device-map auto \
  --max-gpu-memory 7GiB \
  --overwrite
```

Supplemental same-bit-width INT4_AWQ comparison:

```bash
docker compose -f docker/compose.yaml run --rm \
  --volume <repo>/Bonus:/workspace/Bonus \
  --entrypoint uv train run python Bonus/bonus2_quant/quantize_modelopt.py \
  --install-missing-modelopt \
  --modelopt-install-spec nvidia-modelopt==0.44.0 \
  --format int4_awq \
  --n-calib 32 \
  --max-seq-len 512 \
  --device-map cuda \
  --report-prefix modelopt_int4 \
  --overwrite
```

Native vLLM smoke:

```bash
docker run --gpus all --rm --ipc=host \
  -v <repo>/Bonus/bonus2_quant:/bonus \
  -v <repo>/tests:/tests \
  --entrypoint /usr/bin/python3 vllm/vllm-openai:v0.10.2 \
  /bonus/smoke_vllm_modelopt.py \
  --model /bonus/artifacts/modelopt_fp8 \
  --prompts /tests/fixtures/quant_probe_prompts.jsonl \
  --baseline /bonus/reports/fp16_probe.jsonl \
  --output /bonus/reports/fp8_vllm_native_smoke.jsonl \
  --gpu-memory-utilization 0.55 \
  --max-model-len 2048 \
  --max-new-tokens 96
```

## Calibration

Calibration reuses `sales_agent.quant.calibration.load_calibration_texts`, so it
matches the M7 stratified rendering approach. The FP8 run used 64 samples:

| Scenario | Samples |
|---|---:|
| general | 39 |
| objection_handling | 11 |
| info_gathering | 8 |
| closing | 3 |
| cold_open | 3 |

The rendered text is plain ChatML from the training domain. It does not include
extra synthetic prompt templates beyond the existing M7 calibration renderer.

The supplemental INT4_AWQ run used 32 samples to fit a full CUDA load and avoid
the `device_map=auto` meta-tensor export failure:

| Scenario | Samples |
|---|---:|
| general | 19 |
| objection_handling | 5 |
| info_gathering | 4 |
| closing | 2 |
| cold_open | 2 |

## Size And Method Comparison

The FP8 row below is a cross-recipe hardware-path comparison. It shows that the
Ada RTX 4070 can run a ModelOpt FP8 checkpoint natively in vLLM, but it should
not be read as a fair size comparison with INT4.

| Path | Tool | Method | Size | Compression | Smoke quality |
|---|---|---|---:|---:|---|
| FP16 merged | transformers | BF16/FP16 dense HF checkpoint | 8.045 GB | 1.00x | Baseline |
| M7 AWQ | llm-compressor | W4A16_ASYM, group 128, AWQ | 2.666 GB | 3.018x | 5 FP16-vs-INT4 probes showed no obvious degradation |
| Bonus 2 FP8 | ModelOpt | FP8 W8A8, `lm_head` excluded | 4.412 GB | 1.824x | 5 native vLLM probes passed |

The fair same-bit-width size comparison is:

| INT4 path | Tool | Export format | Size | Compression | Serving status |
|---|---|---|---:|---:|---|
| M7 AWQ | llm-compressor | compressed-tensors W4A16_ASYM/g128 | 2.666 GB | 3.018x | PASS in vLLM v0.10.2 |
| Bonus 2 INT4_AWQ | ModelOpt | ModelOpt W4A16_AWQ/g128 | 2.710 GB | 2.969x | FAIL in vLLM v0.10.2; unsupported ModelOpt INT4 format |

Interpretation:

- M7 AWQ is the smaller deployment artifact and remains the best 12GB serving
  choice for this project.
- ModelOpt FP8 is less compressed but uses a floating-point 8-bit format with
  native Ada Tensor Core support, making it the more hardware-specific advanced
  comparison path.
- ModelOpt INT4_AWQ is close in size to M7 AWQ, so it is the useful
  apples-to-apples size comparison. Its current blocker is serving support:
  vLLM accepts ModelOpt FP8/NVFP4 but rejected ModelOpt W4A16_AWQ.
- This ModelOpt run did not quantize the KV cache (`kv_cache_quant_algo=null`),
  because the release recipe registry path for FP8 KV cast was unavailable in
  the wheel used here.

## Smoke Test Context

The project already uses smoke tests throughout M1-M11 rather than a separate
framework:

- M1-M5 scripts expose `--smoke` to validate data/API/GPU pipeline wiring with
  tiny sample counts or step counts.
- M6 merge runs an 8-prompt consistency check between PEFT adapter inference and
  the merged dense model.
- M7 AWQ runs a load/generate self-check and a fixed 5-prompt FP16-vs-INT4 probe.
- M8 serving starts vLLM, polls `/health`, and runs a small OpenAI-compatible
  generation demo.
- M9/M10/M11 reuse fixed samples or short runs for generation/eval/bench smoke,
  with full reports kept separately under `reports/`.

For Bonus 2, the same M7 5-prompt probe set was reused:

| Probe | Native vLLM FP8 result |
|---|---|
| objection | Semantically similar; keeps price objection handling and value framing. |
| info | Exact same response as FP16 baseline. |
| pricing | Exact same response as FP16 baseline; no invented per-kWh rate. |
| cold_open | Similar cold open; no load failure or malformed output. |
| closing | Exact same response as FP16 baseline. |

The supplemental INT4_AWQ fake-quant probe also generated 5/5 non-empty
responses. It stayed reasonable on objection/info/pricing/closing and did not
invent concrete rates, but the cold-open probe drifted from energy sales into a
generic digital-marketing cold open. That is recorded as a quality caveat rather
than a pass equivalent to M7.

Native vLLM log evidence:

- vLLM image: `vllm/vllm-openai:v0.10.2`
- detected checkpoint: `quantization=modelopt`
- model load: `4.2292 GiB`
- KV cache memory: `1.61 GiB`
- output throughput during 5-probe smoke: about `175 output tok/s`

The train container is based on a CUDA runtime image and does not include `nvcc`,
so ModelOpt's optional simulated FP8 CUDA extension could not be built during
the fake-quant probe. That limitation did not block export, and the native vLLM
smoke is the deployment-relevant check.

The native vLLM check for the ModelOpt INT4_AWQ artifact failed before
generation with:

```text
ModelOpt currently only supports: ['FP8', 'NVFP4'] quantizations in vLLM.
```

That confirms the distinction: ModelOpt INT4_AWQ gives a fair size comparison,
but M7's llm-compressor compressed-tensors AWQ remains the deployable INT4 path.

## Hardware Boundary

The RTX 4070 is Ada (`sm89`), confirmed inside Docker by
`torch.cuda.get_device_capability(0) == (8, 9)`. Ada's fourth-generation Tensor
Cores support FP8. That is the real hardware advantage used by this Bonus path.

A100 is Ampere (`sm80`) and does not have native FP8 Tensor Core support. It is
excellent for BF16/TF32/FP16/INT8/INT4, but FP8 is not its native advantage.

NVFP4 is intentionally out of scope here. It is a Blackwell-era path, so this
Bonus implementation stops at FP8 and only falls back to ModelOpt INT4 if FP8 is
unavailable.

## Reproducibility Notes

- The ModelOpt manifest has `git_commit: null` because the train compose service
  mounts selected project directories but not `.git`.
- The large checkpoint is intentionally not tracked.
- The vLLM native smoke output is tracked and small:
  `reports/fp8_vllm_native_smoke.jsonl`.
