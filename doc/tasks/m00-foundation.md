# M0 Engineering Foundation — Task List

> English snapshot of `doc/tasks/m00-foundation.md`. The Chinese file is the source of
> truth; if the two diverge, the Chinese original wins. Translated for the public repo (M12 delivery).

> Prerequisites: none (foundation for all modules)

## Tasks

- [x] T0.1 Initialize the project structure: per the project layout convention, create the directory skeleton `src/sales_agent/`, `scripts/`, `configs/`, `tests/`, `docker/`, `data/`, `models/`, `reports/` (incl. `__init__.py`; `.gitignore` ignores large files under data/models/reports)
- [x] T0.2 Configure `pyproject.toml`: uv dependency groups `dev` / `api` / `bench` (the `gpu` group is declared only, installed inside the container); `uv sync` passes; ruff + pytest base config
- [x] T0.3 Implement `common/schema.py`: `DialogueRecord`, `PreferencePair` Pydantic models + `validate_dialogue()` semantic checks (role alternation, system first, n_turns consistency, lang)
- [x] T0.4 Author `tests/fixtures/`: 5–10 samples per contract (valid + boundary-invalid: out-of-order roles, empty content, missing fields); all `tests/test_schema.py` assertions pass
- [x] T0.5 Implement `common/config.py` (YAML loading + relative-path resolution + seed) and `common/manifest.py` (input hashes, config snapshot, git commit, timestamp)
- [x] T0.6 Implement `common/openrouter.py`: OpenAI SDK pointed at OpenRouter, concurrency semaphore, exponential-backoff retry, token-usage accounting; `tests/conftest.py` provides a programmable `fake_openrouter` mock fixture
- [x] T0.7 Author `docker/Dockerfile.train` and `compose.yaml` (train/serve services, volume mounts, GPU passthrough)
- [x] T0.8 Verify container GPU access: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` prints the RTX 4070 (**highest-priority Day 1 risk item**)

## Definition of Done (DoD)

`uv run pytest` green; `nvidia-smi` works inside the container; schema/fixtures/mock are directly reusable by later module tests.
