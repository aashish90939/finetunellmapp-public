# LLM Finetuning Workbench

This repo provides a FastAPI + React UI for fine-tuning (or just running) HF causal LMs with LoRA/QLoRA, plus an inference playground. Core training/inference logic lives in `app_core/` and is exposed via backend APIs and the frontend UI.

## Why this project is useful

This project is a practical control center for fine-tuning workflows:

- Configure training/inference settings from a UI instead of editing scripts each time.
- Launch runs quickly with dataset upload + config overrides.
- Monitor progress, loss, and logs in one place.
- Reuse finished runs for inference without manual wiring.

In short: less setup friction, better visibility, faster iteration.

## Who this is for

- Builders who want a usable fine-tuning dashboard, not just raw training scripts.
- Teams/solo devs who run experiments and need run tracking and repeatability.
- People hosting model training on a server but wanting a simple web UI to control it.
- Learners who want to understand LoRA/QLoRA workflows end-to-end.

## Where to run it

- Recommended: run backend training on Linux + Nvidia GPU (cloud or on-prem), and use the UI to manage runs.
- Optional: run everything locally for lightweight experiments if your machine has enough memory/compute.
- Mac laptops are good for development, API/UI testing, and small experiments; serious fine-tuning is usually better on GPU servers.

## Hardware guide (rough)

These numbers are practical starting points for LoRA/QLoRA-style fine-tuning and can vary by sequence length, batch size, and model architecture.

| Model size | Local Mac (unified memory)          | Nvidia GPU (VRAM) | Practical expectation               |
| ---------- | ----------------------------------- | ----------------- | ----------------------------------- |
| 1B to 1.5B | 16GB minimum, 24GB smoother         | 8GB to 16GB       | Good for small/medium experiments   |
| 3B to 4B   | 24GB to 32GB preferred              | 16GB to 24GB      | Usually better on GPU server        |
| 7B to 8B   | 48GB+ recommended                   | 24GB to 48GB      | Server-first, local is limited/slow |
| 13B+       | Not practical for most local setups | 48GB+             | Use dedicated GPU infrastructure    |

If running locally with low memory, start with:

- `model_name` in the 1B to 1.5B range
- `per_device_train_batch_size=1`
- shorter sequence lengths (`max_source_length`/`max_target_length`)
- `load_in_4bit=False` on Mac when bitsandbytes is unavailable

## Stack

- Backend: FastAPI, `transformers`, `peft`, optional bitsandbytes (4-bit), PyTorch.
- Frontend: React + TypeScript (Vite).
- Training wrapper: `shared/training/run.py` wraps `app_core` modules for run-scoped outputs.

## Repo layout

- `app_core/`: core training/inference modules (`config.py`, `model.py`, `prompts.py`, etc.).
- `shared/training/run.py`: programmatic training entry that creates per-run folders, logs, metrics.
- `backend/`: FastAPI app and services (`/api/config`, `/api/datasets`, `/api/runs`, `/api/infer`, `/api/auth`).
- `frontend/`: React UI (forms for config, dataset upload, run launch, logs/metrics, inference).
- `docker-compose.yml`: spins up backend + frontend.

## Running (Docker)

> Mac users: CUDA is not available; you’ll run CPU-only. On M1/M2, add `platform: linux/amd64` under the backend service if you see arch mismatch warnings (slower under emulation), or run native Arm with the CPU base image (default).

1. Build & run (CPU default):

   ```bash
   docker-compose build
   docker-compose up
   ```

   Backend: http://localhost:8000, Frontend: http://localhost:4173

2. Use a CUDA base (Linux + Nvidia):

   ```bash
   BASE_IMAGE=pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime docker-compose build --no-cache backend
   docker-compose up
   ```

   Ensure the host has Nvidia drivers and set GPU access in compose if desired.

3. Cache HF models: mount the HF cache to persist downloads:
   ```yaml
   backend:
     volumes:
       - ./hf-cache:/root/.cache/huggingface/hub
   ```

## Recommended deployment pattern

For practical use, treat this repo as a UI + API layer over compute:

- Frontend: local machine or static host.
- Backend: GPU server (RunPod/AWS/GCP/etc.) with persistent storage.
- Volumes: persist `outputs/`, `data/`, and HF cache.

## Running (venv, no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload
cd frontend && npm install && npm run dev
```

## Per-device guidance

- **Mac (M1/M2)**: CPU-only. Set `load_in_4bit=False` (bitsandbytes often unavailable on macOS/MPS). Use smaller models/shorter seq lengths/batch sizes. Optional: `platform: linux/amd64` in compose if you need x86 (slower).
- **Linux + Nvidia GPU**: Use CUDA base image, keep `load_in_4bit=True` for QLoRA, tune batch/sequence lengths to VRAM.
- **Windows**: Use WSL2; follow Linux guidance. Native bitsandbytes support is limited.

## Hugging Face token

- Set `HF_TOKEN` env (Docker or venv), or enter it in the UI “Access” section. Backend stores it in-process and sets `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN`.

## Modes

- **Finetune**: Upload/select a dataset (CSV with `input_text`/`output_text`), tweak config, launch a run. Artifacts per run: `outputs/<run_id>/adapter`, `config.json`, `analysis/learning_curve.png`, `train.log`, `training_history.csv`.
- **Plain model only**: Tick “Plain model only” in the UI and set `model_name`; no dataset needed. The model is registered for inference without training.

## Live progress & logs

- Backend tracks live `progress` (step/epoch/logs) in-memory and writes JSON lines to `outputs/<run_id>/train.log`.
- API: `/api/runs` (list with progress), `/api/runs/{id}` (status), `/api/runs/{id}/log` (tail log).
- Frontend shows current step/loss and lets you view the tail of the run log.

## Key API endpoints

- `GET /api/config/defaults` – current `TrainConfig` defaults.
- `POST /api/auth/token` – set HF token.
- `POST /api/datasets/upload` – upload CSV.
- `POST /api/runs` – start finetune (with `dataset_path`) or plain model (empty `dataset_path`, provide `config_overrides.model_name`).
- `GET /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/log`.
- `POST /api/infer` – inference using a finished run (or plain model registration).

## Frontend controls (what they do)

- **Model (HF id or local path)**: Which base model to load; for plain mode this is what serves; for finetune it is the starting checkpoint.
- **Prompt style**: `auto` infers Mistral/LLaMA/plain; `mistral`, `llama`, `plain` force templates.
- **Max source/target length**: Truncation limits for input/output tokens; longer uses more memory.
- **System prompt**: Freeform task definition; UI shows a neutral placeholder and does not prefill the demo prompt.
- **Dataset upload/path**: CSV with `input_text`/`output_text`. Skipped when “Plain model only” is checked.
- **HF token**: Set a token for gated models; stored in-process only.
- **Plain model only (no finetune)**: Registers the base model for inference; no training, no dataset required.
- **Epochs / LR / Batch size / LoRA r/alpha/dropout / 4-bit toggle**: Training hyperparameters; higher batch/lengths need more memory; disable 4-bit on Mac/CPU if unsupported.
- **Run id (optional)**: Name a run; otherwise auto-generated.
- **Runs list**: Shows status, step/loss (live), adapter path; “View log” tails `train.log`; “Use for inference” selects run for the playground.
- **Inference playground**: Send text to a finished run or plain model; you can still tweak generation params via the API (payload supports temperature/top_p/max_new_tokens).

## Tuning tips

- Adjust `model_name`, `max_source_length`, `max_target_length`, `per_device_train_batch_size`, `gradient_accumulation_steps`, and LoRA params based on memory.
- For Mac/CPU: smaller models (e.g., 1–3B), disable 4-bit, keep sequences short.
- For GPU: keep 4-bit enabled and start with small batch size to avoid OOM.
