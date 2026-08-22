#!/usr/bin/env python3
"""
Standalone Minimal Production Server for Azure VM.
Inference only — no training, no dev packages required.
"""

import os
import time
import uuid
from typing import List, Optional
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 1. API Key from environment or default
API_KEY = os.environ.get("PORTFOLIO_API_KEY", "kj_live_sec_789f2a4b1c")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "checkpoints/mlx_karthik_adapters_1.5b")

app = FastAPI(title="Karthik Jayan Portfolio AI (Azure VM)")

# 2. CORS for karthikjayan.dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://karthikjayan.dev", "https://www.karthikjayan.dev", "https://ridge.karthikjayan.tech", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Model Engine Loader
ENGINE = None
SYSTEM_PROMPT = (
    "You are Karthik Jayan, an AI Systems Engineer. Always answer in the first person as Karthik. "
    "When asked about 'Ridge', 'Recall', 'CyberLabs', or any projects, refer to your software engineering systems (e.g., Ridge is your Self-Correcting RAG platform at ridge.karthikjayan.tech), never generic dictionary or geological definitions. "
    "Stay strictly grounded in your verified experience, projects, and background from karthikjayan.dev."
)


def get_model():
    global ENGINE
    if ENGINE is None:
        print("⚡ Loading inference engine on Azure VM...")
        try:
            # Check if running on macOS with Apple MLX
            import mlx.core
            from mlx_lm import load
            model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit", adapter_path=ADAPTER_PATH)
            ENGINE = ("mlx", model, tokenizer)
            print("✅ Apple MLX Engine loaded.")
        except Exception:
            # Running on Linux Azure VM (PyTorch / Transformers)
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("✅ Loading PyTorch / Transformers Engine on Linux...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
            if device == "cpu":
                model = model.to("cpu")
            ENGINE = ("hf", model, tokenizer)
            print(f"✅ Loaded on {device.upper()} ({dtype}).")
    return ENGINE


# 4. Auth Dependency
def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    key = x_api_key or (authorization.split("Bearer ")[1].strip() if authorization and authorization.startswith("Bearer ") else None)
    if not key or key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key.")
    return key


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 300


@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest, api_key: str = Depends(verify_api_key)):
    engine_type, model, tokenizer = get_model()
    start_t = time.perf_counter()

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in req.messages:
        if m.role != "system":
            msgs.append({"role": m.role, "content": m.content})

    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    if engine_type == "mlx":
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=req.temperature, top_p=0.9, top_k=40)
        output = generate(model, tokenizer, prompt=formatted, max_tokens=req.max_tokens, sampler=sampler)
    else:
        import torch
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=max(req.temperature, 0.01),
                do_sample=True
            )
        output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    elapsed_s = time.perf_counter() - start_t
    gen_tokens = len(tokenizer.encode(output))

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
        "object": "chat.completion",
        "model": "karthik-qwen2.5-1.5b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": output.strip()}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": gen_tokens, "total_tokens": len(tokenizer.encode(formatted)) + gen_tokens},
        "performance": {"time_ms": elapsed_s * 1000, "tokens_per_sec": gen_tokens / max(elapsed_s, 1e-4)}
    }


@app.post("/api/chat")
def simple_chat(req: ChatRequest, api_key: str = Depends(verify_api_key)):
    res = chat_completions(req, api_key=api_key)
    return {
        "reply": res["choices"][0]["message"]["content"],
        "tokens": res["usage"]["completion_tokens"],
        "time_ms": res["performance"]["time_ms"],
        "tok_per_sec": res["performance"]["tokens_per_sec"]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Azure VM Server on port {port}...")
    uvicorn.run("serve:app", host="0.0.0.0", port=port, workers=1)
