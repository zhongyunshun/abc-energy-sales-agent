# ModelOpt INT4_AWQ Native vLLM Smoke

Status: FAIL, unsupported by vLLM v0.10.2.

The checkpoint exported successfully and has this ModelOpt quantization config:

```json
{
  "quant_algo": "W4A16_AWQ",
  "kv_cache_quant_algo": null,
  "group_size": 128,
  "has_zero_point": false,
  "pre_quant_scale": true,
  "exclude_modules": ["lm_head"]
}
```

Native vLLM loading failed before generation with:

```text
ModelOpt currently only supports: ['FP8', 'NVFP4'] quantizations in vLLM.
Please check the `hf_quant_config.json` file for your model's quant configuration.
```

Conclusion: ModelOpt INT4_AWQ is useful here as a same-bit-width size and
fake-quant quality comparison against M7, but it is not a drop-in vLLM serving
artifact in the current vLLM version. The M7 llm-compressor compressed-tensors
AWQ checkpoint remains the deployable INT4 path.
