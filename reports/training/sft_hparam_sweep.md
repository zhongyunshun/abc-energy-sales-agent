# M4 SFT hyperparameter sweep

Base config: `configs/sft_a100.yaml` | budget: 0.4 epoch(s)/combo | alpha = 2*rank | torch.compile off (HP-orthogonal).

| LoRA r | alpha | lr | steps | best val loss | status |
|---|---|---|---|---|---|
| 32 | 64 | 2e-04 | 294 | 0.9496 | ok |
| 16 | 32 | 2e-04 | 294 | 0.9720 | ok |
| 32 | 64 | 1e-04 | 294 | 0.9831 | ok |
| 16 | 32 | 1e-04 | 294 | 1.0121 | ok |
