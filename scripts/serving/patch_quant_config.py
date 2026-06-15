"""M8 serve-compat shim: drop quant-config keys that the pinned vLLM rejects.

Why this exists (real-machine finding, 2026-06-14): M7's AWQ product is written by
llm-compressor / compressed-tensors 0.16.0, whose per-group weights config carries
`scale_dtype` and `zp_dtype` (dtype hints for the scale / zero-point tensors). The
pinned serving image (vllm/vllm-openai:v0.10.2) parses quantization_config with a
pydantic model set to extra='forbid', so those two newer fields crash startup
("Extra inputs are not permitted"). They are pure metadata -- vLLM's W4A16 kernel
reads the actual tensor dtypes from the safetensors -- so removing them lets v0.10.2
load the model unchanged. pydantic reports ALL extra fields at once and flagged only
these two, so stripping them is sufficient, not just necessary.

This is the same pattern M7 already uses (_drop_transformers_version in
quantize_awq.py): an idempotent config.json metadata patch for a downstream loader.
serve.sh runs this before `docker compose up serve`. Idempotent: a second run finds
nothing to strip and exits 0.

Exit codes (the CLI contract): 0 success (patched or already clean), 2 the model
config.json is missing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sales_agent.common.config import load_config

logger = logging.getLogger("patch_quant_config")

EXIT_OK = 0
EXIT_CONTRACT = 2

# Keys that compressed-tensors >= 0.16.0 emits but vLLM v0.10.2 forbids.
INCOMPATIBLE_QUANT_KEYS = ("scale_dtype", "zp_dtype")


def strip_keys(node: object, keys: tuple[str, ...], path: str = "") -> list[str]:
    """Recursively remove keys named in ``keys`` from a nested dict/list (in place).

    Returns the dotted paths removed. Pure logic (no I/O) -- unit-tested for
    idempotency and for leaving every other field untouched.
    """
    removed: list[str] = []
    if isinstance(node, dict):
        for k in list(node.keys()):
            cur = f"{path}.{k}" if path else k
            if k in keys:
                node.pop(k)
                removed.append(cur)
            else:
                removed.extend(strip_keys(node[k], keys, cur))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            removed.extend(strip_keys(item, keys, f"{path}[{i}]"))
    return removed


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/serve.yaml")
    parser.add_argument("--model-dir", default=None, help="override config model.dir (host path)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    model_dir = Path(args.model_dir or cfg["model"]["dir"])
    config_path = model_dir / "config.json"
    if not config_path.exists():
        logger.error("model config.json not found at %s (run M7 first?)", config_path)
        return EXIT_CONTRACT

    data = json.loads(config_path.read_text(encoding="utf-8"))
    qc = data.get("quantization_config")
    if not isinstance(qc, dict):
        logger.info("no quantization_config in %s; nothing to patch", config_path)
        return EXIT_OK

    removed = strip_keys(qc, INCOMPATIBLE_QUANT_KEYS)
    if not removed:
        logger.info("quantization_config already vLLM-v0.10.2 compatible (no keys stripped)")
        return EXIT_OK

    config_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(
        "stripped %d vLLM-incompatible key(s) from %s: %s",
        len(removed), config_path, ", ".join(f"quantization_config.{p}" for p in removed),
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
