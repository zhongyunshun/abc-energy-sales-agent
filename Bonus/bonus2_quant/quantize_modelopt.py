"""Bonus 2: NVIDIA ModelOpt PTQ path for the merged FP16 sales-agent model.

This script intentionally lives under ``Bonus/bonus2_quant`` and does not modify
the main M7 quantization code. It uses the existing project calibration sampler
(``sales_agent.quant.calibration``) and writes all large artifacts under this
Bonus directory, which is gitignored.

Run inside the train Docker container, for example:

    docker compose -f docker/compose.yaml run --rm \
      --volume <repo>/Bonus:/workspace/Bonus \
      --entrypoint uv train run python Bonus/bonus2_quant/quantize_modelopt.py \
      --install-missing-modelopt --format fp8
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sales_agent.common.io import read_jsonl
from sales_agent.common.manifest import build_manifest, write_manifest
from sales_agent.common.schema import Message
from sales_agent.quant.calibration import (
    load_calibration_texts,
    model_dir_size_bytes,
    size_report,
)
from sales_agent.training.formatting import render_chatml

LOGGER = logging.getLogger("bonus2_modelopt")

REPO_ROOT = Path(__file__).resolve().parents[2]
BONUS_ROOT = REPO_ROOT / "Bonus" / "bonus2_quant"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "adapters" / "merged"
DEFAULT_CALIBRATION_PATH = REPO_ROOT / "data" / "processed" / "train.jsonl"
DEFAULT_PROBES_PATH = REPO_ROOT / "tests" / "fixtures" / "quant_probe_prompts.jsonl"
DEFAULT_REPORTS_DIR = BONUS_ROOT / "reports"
DEFAULT_ARTIFACTS_DIR = BONUS_ROOT / "artifacts"
MODEL_OPT_MIN_DEPS = [
    "ninja",
    "omegaconf>=2.3.0",
    "pulp<4.0",
    "rich",
    "setuptools>=80",
]

EXIT_OK = 0
EXIT_CONTRACT = 2
EXIT_DEPENDENCY = 3


def _install_modelopt_if_requested(install: bool, spec: str, no_deps: bool) -> None:
    """Import ModelOpt, optionally installing it into the current container first."""
    try:
        importlib.import_module("modelopt")
        return
    except ImportError:
        if not install:
            raise

    LOGGER.info("modelopt is missing; installing %s in this container", spec)
    uv = shutil.which("uv")
    install_args = ["install", "--python", sys.executable, "-U"]
    if no_deps:
        install_args.append("--no-deps")
    install_args.append(spec)
    if uv is not None:
        subprocess.check_call([uv, "pip", *install_args])
        if no_deps:
            subprocess.check_call(
                [uv, "pip", "install", "--python", sys.executable, "-U", *MODEL_OPT_MIN_DEPS]
            )
    else:
        subprocess.check_call(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
        )
        pip_args = ["install", "-U"]
        if no_deps:
            pip_args.append("--no-deps")
        pip_args.append(spec)
        subprocess.check_call([sys.executable, "-m", "pip", *pip_args])
        if no_deps:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-U", *MODEL_OPT_MIN_DEPS]
            )
    importlib.invalidate_caches()
    importlib.import_module("modelopt")


def _safe_reset_dir(path: Path) -> None:
    """Remove and recreate an output directory, constrained to Bonus artifacts."""
    path = path.resolve()
    allowed = (DEFAULT_ARTIFACTS_DIR.resolve(), DEFAULT_REPORTS_DIR.resolve())
    if not any(path == root or root in path.parents for root in allowed):
        raise ValueError(f"refusing to reset non-Bonus output directory: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _weight_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*.safetensors"))


def _torch_dtype(name: str):
    import torch

    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as e:
        raise ValueError(f"unsupported dtype {name!r}") from e


def _input_device(model) -> str:
    """Choose a device for input tensors when a HF device map may be present."""
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for device in device_map.values():
            if isinstance(device, int):
                return f"cuda:{device}"
            if isinstance(device, str) and device not in {"cpu", "disk", "meta"}:
                return device
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return str(device)
    return "cuda:0"


def _load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs: dict[str, Any] = {
        "dtype": _torch_dtype(args.dtype),
        "low_cpu_mem_usage": True,
    }
    if args.device_map == "cuda":
        model = AutoModelForCausalLM.from_pretrained(str(args.model_dir), **kwargs).to("cuda")
    else:
        kwargs["device_map"] = args.device_map
        kwargs["max_memory"] = {0: args.max_gpu_memory, "cpu": args.max_cpu_memory}
        kwargs["offload_folder"] = str(args.offload_dir)
        model = AutoModelForCausalLM.from_pretrained(str(args.model_dir), **kwargs)
    model.eval()
    torch.cuda.empty_cache()
    return model, tokenizer


def _iter_tokenized_batches(
    tokenizer,
    texts: list[str],
    max_seq_len: int,
    device: str,
) -> Iterator[dict[str, Any]]:
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        yield {k: v.to(device) for k, v in encoded.items()}


def _make_forward_loop(tokenizer, texts: list[str], max_seq_len: int, device: str):
    import torch

    def forward_loop(model) -> None:
        with torch.no_grad():
            for batch in _iter_tokenized_batches(tokenizer, texts, max_seq_len, device):
                model(**batch)

    return forward_loop


def _load_quant_config(format_name: str, recipe: str | None) -> tuple[dict[str, Any], str]:
    if format_name == "fp8":
        import modelopt.torch.quantization as mtq

        recipe_name = recipe or "general/ptq/fp8_default-kv_fp8_cast"
        try:
            from modelopt.recipe import load_recipe

            loaded = load_recipe(recipe_name)
            return loaded.quantize.model_dump(), recipe_name
        except Exception as e:  # noqa: BLE001 -- release wheels may not ship recipe files
            LOGGER.warning(
                "could not load ModelOpt recipe %s (%s); falling back to mtq.FP8_DEFAULT_CFG",
                recipe_name,
                e,
            )
            return copy.deepcopy(mtq.FP8_DEFAULT_CFG), "mtq.FP8_DEFAULT_CFG"

    import modelopt.torch.quantization as mtq

    if format_name == "int4_awq":
        cfg = copy.deepcopy(mtq.INT4_AWQ_CFG)
        cfg["quant_cfg"].append({"quantizer_name": "*lm_head*", "enable": False})
        return cfg, "mtq.INT4_AWQ_CFG"

    raise ValueError(f"unsupported format: {format_name}")


def _render_probe_prompts(path: Path) -> list[dict[str, str]]:
    prompts = []
    for raw in read_jsonl(path):
        messages = [Message(**m) for m in raw["context"]]
        prompts.append(
            {
                "id": raw["id"],
                "prompt": render_chatml(messages, add_generation_prompt=True),
            }
        )
    return prompts


def _generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    device = _input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _run_fakequant_probe(
    model,
    tokenizer,
    probes_path: Path,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    probes = _render_probe_prompts(probes_path)
    rows = []
    for probe in probes:
        rows.append(
            {
                "id": probe["id"],
                "output": _generate_one(model, tokenizer, probe["prompt"], max_new_tokens),
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def _report_paths(reports_dir: Path, prefix: str, format_name: str) -> dict[str, Path]:
    """Stable report names; keep the original FP8 names for the default prefix."""
    if prefix == "modelopt":
        return {
            "summary": reports_dir / ".quant_summary.txt",
            "manifest": reports_dir / "modelopt_manifest.json",
            "baseline": reports_dir / "fp16_probe.jsonl",
            "quant": reports_dir / f"{format_name}_fakequant_probe.jsonl",
            "markdown": reports_dir / "modelopt_quant_report.md",
        }
    return {
        "summary": reports_dir / f"{prefix}_quant_summary.txt",
        "manifest": reports_dir / f"{prefix}_manifest.json",
        "baseline": reports_dir / f"{prefix}_fp16_probe.jsonl",
        "quant": reports_dir / f"{prefix}_{format_name}_fakequant_probe.jsonl",
        "markdown": reports_dir / f"{prefix}_quant_report.md",
    }


def _write_markdown_report(
    path: Path,
    manifest: dict[str, Any],
    baseline_rows: list[dict[str, str]],
    quant_rows: list[dict[str, str]],
) -> None:
    stats = manifest["stats"]
    sizes = stats["sizes"]
    lines = [
        "# Bonus 2 ModelOpt Quantization Run",
        "",
        f"- format: `{stats['format']}`",
        f"- recipe: `{stats['recipe']}`",
        f"- model: `{stats['model_dir']}`",
        f"- artifact: `{stats['output_dir']}`",
        f"- calibration: {stats['calibration']['n_selected']} samples, "
        f"max_seq_len {stats['max_seq_len']}, per-scenario "
        f"{stats['calibration']['per_scenario']}",
        "",
        "## Size",
        "",
        "| Model | GB | Bytes |",
        "|---|---:|---:|",
        f"| FP16 merged | {sizes['fp16_gb']} | {sizes['fp16_bytes']:,} |",
        f"| ModelOpt {stats['format']} | {sizes['quant_gb']} | {sizes['quant_bytes']:,} |",
        "",
        f"- compression ratio: {sizes['compression_ratio']}x",
        f"- size reduction: {sizes['size_reduction_pct']}%",
        "",
        "## Fake-Quant Probe",
        "",
    ]
    quant_by_id = {row["id"]: row["output"] for row in quant_rows}
    for base in baseline_rows:
        lines.extend(
            [
                f"### {base['id']}",
                "",
                f"**FP16:** {base['output'].strip()}",
                "",
                f"**ModelOpt:** {quant_by_id.get(base['id'], '').strip()}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=["fp8", "int4_awq"], default="fp8")
    parser.add_argument("--recipe", default="general/ptq/fp8_default-kv_fp8_cast")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--probes-path", type=Path, default=DEFAULT_PROBES_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--report-prefix", default="modelopt")
    parser.add_argument("--n-calib", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", choices=["auto", "sequential", "cuda"], default="auto")
    parser.add_argument("--max-gpu-memory", default="7GiB")
    parser.add_argument("--max-cpu-memory", default="48GiB")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--install-missing-modelopt", action="store_true")
    parser.add_argument("--modelopt-install-spec", default="nvidia-modelopt")
    parser.add_argument("--modelopt-install-deps", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-fakequant-smoke", action="store_true")
    parser.add_argument(
        "--offload-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR / "_offload",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    args.model_dir = args.model_dir.resolve()
    args.calibration_path = args.calibration_path.resolve()
    args.probes_path = args.probes_path.resolve()
    args.reports_dir = args.reports_dir.resolve()
    args.offload_dir = args.offload_dir.resolve()
    if args.output_dir is None:
        args.output_dir = DEFAULT_ARTIFACTS_DIR / f"modelopt_{args.format}"
    args.output_dir = args.output_dir.resolve()

    if not (args.model_dir / "config.json").exists():
        LOGGER.error("merged model not found at %s", args.model_dir)
        return EXIT_CONTRACT
    if not args.calibration_path.exists():
        LOGGER.error("calibration source not found at %s", args.calibration_path)
        return EXIT_CONTRACT
    if not args.probes_path.exists():
        LOGGER.error("probe fixture not found at %s", args.probes_path)
        return EXIT_CONTRACT
    if args.output_dir.exists() and not args.overwrite:
        LOGGER.error("output dir exists; pass --overwrite to replace it: %s", args.output_dir)
        return EXIT_CONTRACT

    try:
        _install_modelopt_if_requested(
            args.install_missing_modelopt,
            args.modelopt_install_spec,
            no_deps=not args.modelopt_install_deps,
        )
        import torch
        import modelopt.torch.opt as mto
        import modelopt.torch.quantization as mtq
        from modelopt.torch.export import export_hf_checkpoint
    except Exception as e:  # noqa: BLE001
        LOGGER.error("ModelOpt stack unavailable in this container: %s", e)
        return EXIT_DEPENDENCY

    if not torch.cuda.is_available():
        LOGGER.error("CUDA is unavailable; run inside the GPU train container")
        return EXIT_DEPENDENCY

    _safe_reset_dir(args.output_dir)
    args.offload_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    report_paths = _report_paths(args.reports_dir, args.report_prefix, args.format)

    texts, calibration_report = load_calibration_texts(
        args.calibration_path, args.n_calib, args.seed
    )
    LOGGER.info(
        "calibration: %d texts | per-scenario=%s",
        calibration_report.n_selected,
        calibration_report.per_scenario,
    )

    try:
        torch.compiler.set_stance("force_eager")
    except Exception:
        pass
    mto.enable_huggingface_checkpointing()

    model, tokenizer = _load_model(args)
    device = _input_device(model)
    LOGGER.info("model input device: %s", device)

    baseline_rows: list[dict[str, str]] = []
    if not args.skip_fakequant_smoke:
        LOGGER.info("running FP16 baseline probe before quantization")
        baseline_rows = _run_fakequant_probe(
            model, tokenizer, args.probes_path, args.max_new_tokens
        )

    quant_cfg, recipe_label = _load_quant_config(
        args.format, args.recipe if args.format == "fp8" else None
    )
    forward_loop = _make_forward_loop(tokenizer, texts, args.max_seq_len, device)
    LOGGER.info("running ModelOpt quantization: format=%s recipe=%s", args.format, recipe_label)
    with torch.inference_mode():
        model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

    quant_rows: list[dict[str, str]] = []
    if not args.skip_fakequant_smoke:
        LOGGER.info("running ModelOpt fake-quant probe after quantization")
        quant_rows = _run_fakequant_probe(model, tokenizer, args.probes_path, args.max_new_tokens)

    LOGGER.info("exporting unified HF checkpoint to %s", args.output_dir)
    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    summary_dir = args.reports_dir / f"_{args.report_prefix}_quant_summary"
    _safe_reset_dir(summary_dir)
    mtq.print_quant_summary(model, str(summary_dir))
    summary_src = summary_dir / ".quant_summary.txt"
    if summary_src.exists():
        shutil.copy2(summary_src, report_paths["summary"])
    shutil.rmtree(summary_dir, ignore_errors=True)

    fp16_bytes = model_dir_size_bytes(args.model_dir)
    quant_bytes = _weight_size_bytes(args.output_dir)
    base_size = size_report(fp16_bytes, quant_bytes)
    sizes = {
        "fp16_bytes": fp16_bytes,
        "quant_bytes": quant_bytes,
        "fp16_gb": base_size["fp16_gb"],
        "quant_gb": round(quant_bytes / 1e9, 3),
        "compression_ratio": round(fp16_bytes / quant_bytes, 3) if quant_bytes else None,
        "size_reduction_pct": (
            round((1 - quant_bytes / fp16_bytes) * 100, 2) if fp16_bytes else None
        ),
    }
    stats = {
        "format": args.format,
        "recipe": recipe_label,
        "model_dir": str(args.model_dir),
        "output_dir": str(args.output_dir),
        "device_map": args.device_map,
        "max_gpu_memory": args.max_gpu_memory,
        "dtype": args.dtype,
        "n_calib": args.n_calib,
        "max_seq_len": args.max_seq_len,
        "calibration": calibration_report.as_dict(),
        "sizes": sizes,
        "fakequant_smoke": {
            "enabled": not args.skip_fakequant_smoke,
            "probe_count": len(quant_rows),
            "baseline_path": str(report_paths["baseline"]),
            "modelopt_path": str(report_paths["quant"]),
        },
    }
    inputs = sorted(args.model_dir.glob("*.safetensors")) + [args.calibration_path, args.probes_path]
    manifest = build_manifest(
        inputs=inputs,
        config=_jsonable(vars(args)),
        stats=stats,
        repo_root=REPO_ROOT,
    )
    write_manifest(args.output_dir, manifest)
    manifest_path = report_paths["manifest"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if baseline_rows:
        _write_jsonl(report_paths["baseline"], baseline_rows)
    if quant_rows:
        _write_jsonl(report_paths["quant"], quant_rows)
        _write_markdown_report(
            report_paths["markdown"],
            manifest,
            baseline_rows,
            quant_rows,
        )

    LOGGER.info(
        "done: FP16 %.3f GB -> ModelOpt %s %.3f GB (%sx)",
        sizes["fp16_gb"],
        args.format,
        sizes["quant_gb"],
        sizes["compression_ratio"],
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
