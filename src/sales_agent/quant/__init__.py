"""M7 AWQ quantization support: calibration-set construction and size accounting.

The GPU quantization itself (llm-compressor ``oneshot``) lives in the thin CLI
``scripts/quant/quantize_awq.py`` and runs inside the train container; this
package holds only the pure, unit-testable logic (the M7 contract).
"""
