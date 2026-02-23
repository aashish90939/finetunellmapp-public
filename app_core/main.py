# main.py
from __future__ import annotations

from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel
from transformers import TextStreamer

from .config import TrainConfig
from .model import get_peft_model_for_inference, format_instruction


# --------------------------------------------------------------------
# Load fine-tuned model
# --------------------------------------------------------------------

print("[main] Loading fine-tuned model...")

cfg = TrainConfig()
output_dir = Path(cfg.output_dir)

cfg_path = output_dir / "config.json"
if cfg_path.exists():
    # Use the saved config if available
    cfg = TrainConfig.load(cfg_path)

adapter_dir = output_dir / "adapter"
model, tokenizer = get_peft_model_for_inference(cfg, adapter_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("[main] Model loaded on:", device)
streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )



# --------------------------------------------------------------------
# Text generation helper (no streaming, returns pure string)
# --------------------------------------------------------------------

def generate_text(
    input_text: str,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    """
    Generate a single response from the model,
    using the same instruction formatting as during training.
    Returns ONLY the model's output text (no extra labels).
    """
    # Defaults from config if available, otherwise fall back
    if max_new_tokens is None:
        max_new_tokens = getattr(cfg, "gen_max_new_tokens", 2048)
    if temperature is None:
        temperature = getattr(cfg, "gen_temperature", 0.7)
    if top_p is None:
        top_p = getattr(cfg, "gen_top_p", 0.95)

    # Build the prompt using your training-time chat template
    prompt = format_instruction(cfg, input_text)

    # Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Greedy or sampling depending on temperature
    do_sample = temperature is not None and temperature > 0.0
    print(max_new_tokens)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
        )

    # Remove the prompt part and decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    output_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return output_text.strip()


# --------------------------------------------------------------------
# FastAPI app + schemas
# --------------------------------------------------------------------

app = FastAPI(title="LLM LoRA Inference API")

# Allow browser UIs / other clients without CORS issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # if you want to restrict, replace "*" with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    input_text: str
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


# --------------------------------------------------------------------
# 1) API for Postman: pure text in, pure text out
# --------------------------------------------------------------------

@app.post("/api/generate", response_class=PlainTextResponse)
async def api_generate(req: GenerateRequest) -> str:
    """
    Call this from Postman (or curl) with JSON body:
      {
        "input_text": "your text here"
      }

    Response is ONLY the raw model output (text/plain).
    """
    output = generate_text(
        input_text=req.input_text,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    return output


# --------------------------------------------------------------------
# 2) Frontend: simple web UI that calls /api/generate
# --------------------------------------------------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>LLM LoRA Inference</title>
    <style>
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                         sans-serif;
            margin: 0;
            padding: 0;
            background: #0f172a;
            color: #e5e7eb;
        }
        .container {
            max-width: 960px;
            margin: 0 auto;
            padding: 24px 16px 48px;
        }
        h1 {
            font-size: 1.8rem;
            margin-bottom: 0.25rem;
        }
        .subtitle {
            color: #9ca3af;
            margin-bottom: 1.5rem;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        @media (max-width: 900px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
        .card {
            background: #020617;
            border-radius: 14px;
            padding: 16px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.7);
            border: 1px solid #1e293b;
        }
        label {
            font-size: 0.9rem;
            color: #9ca3af;
            display: block;
            margin-bottom: 6px;
        }
        textarea {
            width: 100%;
            min-height: 220px;
            resize: vertical;
            border-radius: 10px;
            border: 1px solid #1f2933;
            background: #020617;
            color: #e5e7eb;
            padding: 10px 12px;
            font-size: 0.95rem;
            outline: none;
            box-sizing: border-box;
        }
        textarea:focus {
            border-color: #4f46e5;
            box-shadow: 0 0 0 1px #4f46e5;
        }
        .output {
            white-space: pre-wrap;
            font-family: system-ui, -apple-system, BlinkMacSystemFont,
                         "Segoe UI", sans-serif;
            font-size: 0.95rem;
        }
        .btn-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 10px;
        }
        button {
            border: none;
            border-radius: 999px;
            padding: 8px 16px;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            background: linear-gradient(135deg, #4f46e5, #6366f1);
            color: white;
        }
        button:disabled {
            opacity: 0.6;
            cursor: default;
        }
        .status {
            font-size: 0.8rem;
            color: #9ca3af;
        }
        .status.error {
            color: #f97373;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>LLM LoRA Inference</h1>
    <div class="subtitle">
        Type your text on the left and see the model output on the right.
    </div>

    <div class="grid">
        <div class="card">
            <label for="input_text">Input text</label>
            <textarea id="input_text"
                      placeholder="Type input text or a question..."></textarea>
            <div class="btn-row">
                <button id="btn_generate">Generate</button>
                <div id="status" class="status"></div>
            </div>
        </div>

        <div class="card">
            <label>Model output</label>
            <div id="output" class="output"></div>
        </div>
    </div>
</div>

<script>
    const inputEl  = document.getElementById("input_text");
    const outputEl = document.getElementById("output");
    const btnEl    = document.getElementById("btn_generate");
    const statusEl = document.getElementById("status");

    async function callApi() {
        const text = inputEl.value.trim();
        if (!text) {
            statusEl.textContent = "Please enter some text first.";
            statusEl.classList.remove("error");
            return;
        }

        btnEl.disabled = true;
        statusEl.textContent = "Generating...";
        statusEl.classList.remove("error");
        outputEl.textContent = "";

        try {
            const resp = await fetch("/api/generate", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    input_text: text
                    // You can also send max_new_tokens / temperature / top_p here if you want
                })
            });

            const data = await resp.text();
            if (!resp.ok) {
                statusEl.textContent = "Error: " + resp.status;
                statusEl.classList.add("error");
            } else {
                statusEl.textContent = "Done.";
                statusEl.classList.remove("error");
                outputEl.textContent = data;
            }
        } catch (err) {
            console.error(err);
            statusEl.textContent = "Request failed. See console.";
            statusEl.classList.add("error");
        } finally {
            btnEl.disabled = false;
        }
    }

    btnEl.addEventListener("click", callApi);
    inputEl.addEventListener("keydown", function (e) {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            callApi();
        }
    });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """
    Frontend: open http://localhost:8000 in your browser.
    """
    return HTML_PAGE


# (Optional) Also expose it explicitly on /ui
@app.get("/ui", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    return HTML_PAGE


# --------------------------------------------------------------------
# Entry point: run with `python main.py`
# --------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # You can change host/port as you like
    uvicorn.run(app, host="0.0.0.0", port=8000)
