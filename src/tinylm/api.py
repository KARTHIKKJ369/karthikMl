"""
Production Headless API Server for Karthik Jayan Portfolio Intelligence.
Supports:
- API Key Authentication (x-api-key or Authorization: Bearer <KEY>)
- OpenAI-compatible endpoint: POST /v1/chat/completions
- Direct Chat endpoint: POST /api/chat
- CORS configuration for karthikjayan.dev and portfolio apps
- Streaming and Non-Streaming responses
- Multi-engine: Apple Silicon MLX (local) & Transformers / PyTorch (Linux Azure VM)
"""

import os
import time
import uuid
from typing import List, Optional, Union
from fastapi import FastAPI, Header, HTTPException, Depends, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Load API Key from environment or default secure fallback
API_KEY = os.environ.get("PORTFOLIO_API_KEY", "kj_live_sec_789f2a4b1c")

app = FastAPI(
    title="Karthik Jayan · Portfolio AI Engine",
    description="Headless API powering portfolio chatbot interactions across karthikjayan.dev",
    version="1.0.0"
)

# Enable CORS for portfolio domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://karthikjayan.dev",
        "https://www.karthikjayan.dev",
        "https://ridge.karthikjayan.tech",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine state
ENGINE = None
SYSTEM_PROMPT = (
    "You are Karthik Jayan, an AI Systems Engineer. Always answer in the first person as Karthik. "
    "When asked about 'Ridge', 'Recall', 'CyberLabs', or any projects, refer to your software engineering systems (e.g., Ridge is your Self-Correcting RAG platform at ridge.karthikjayan.tech), never generic dictionary or geological definitions. "
    "Stay strictly grounded in your verified experience, projects, and background from karthikjayan.dev."
)


def get_engine():
    global ENGINE
    if ENGINE is None:
        # Check if Apple MLX is available
        try:
            import mlx.core
            from mlx_lm import load
            print("🚀 Loading model using Apple MLX engine...")
            model, tokenizer = load(
                "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                adapter_path="checkpoints/mlx_karthik_adapters_1.5b"
            )
            ENGINE = ("mlx", model, tokenizer)
        except Exception as e:
            print(f"⚠️ MLX not available ({e}). Falling back to Hugging Face / Transformers...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            model_id = "Qwen/Qwen2.5-1.5B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            ENGINE = ("hf", model, tokenizer)
    return ENGINE


# Security dependency
def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None, alias="Authorization")
):
    provided_key = None
    if x_api_key:
        provided_key = x_api_key
    elif authorization and authorization.startswith("Bearer "):
        provided_key = authorization.split("Bearer ")[1].strip()

    if not provided_key or provided_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide 'x-api-key: YOUR_KEY' or 'Authorization: Bearer YOUR_KEY'."
        )
    return provided_key


# Request / Response Schemas
class Message(BaseModel):
    role: str = Field(..., description="user, assistant, or system")
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 300
    stream: Optional[bool] = False


class SimpleGenerateRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.2
    max_new_tokens: Optional[int] = 300


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "engine": "mlx" if ENGINE and ENGINE[0] == "mlx" else "transformers",
        "model": "Qwen2.5-1.5B-Instruct-4bit (Fine-Tuned)",
        "timestamp": time.time()
    }


@app.post("/v1/chat/completions")
def openai_chat_completions(
    req: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    OpenAI-compatible Chat Completion API.
    Can be called directly using the official `openai` npm/python package in frontend code.
    """
    engine_type, model, tokenizer = get_engine()
    start_t = time.perf_counter()

    # Prepend System persona if not provided
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in req.messages:
        if m.role != "system":
            msgs.append({"role": m.role, "content": m.content})

    if engine_type == "mlx":
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        sampler = make_sampler(temp=req.temperature, top_p=req.top_p, top_k=40)
        output_text = generate(model, tokenizer, prompt=formatted, max_tokens=req.max_tokens, sampler=sampler)
    else:
        import torch
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=max(req.temperature, 0.01),
                do_sample=True
            )
        output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    elapsed_s = time.perf_counter() - start_t
    generated_tokens = len(tokenizer.encode(output_text))

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "karthik-qwen2.5-1.5b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text.strip()
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(tokenizer.encode(formatted)),
            "completion_tokens": generated_tokens,
            "total_tokens": len(tokenizer.encode(formatted)) + generated_tokens
        },
        "performance": {
            "time_ms": elapsed_s * 1000,
            "tokens_per_sec": generated_tokens / max(elapsed_s, 1e-4)
        }
    }


@app.post("/api/chat")
def simple_chat(
    req: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Standard lightweight REST endpoint for custom portfolio integrations.
    """
    res = openai_chat_completions(req, api_key=api_key)
    return {
        "reply": res["choices"][0]["message"]["content"],
        "tokens": res["usage"]["completion_tokens"],
        "time_ms": res["performance"]["time_ms"],
        "tok_per_sec": res["performance"]["tokens_per_sec"]
    }


@app.post("/api/generate")
def direct_generate(
    req: SimpleGenerateRequest,
    api_key: str = Depends(verify_api_key)
):
    chat_req = ChatRequest(messages=[Message(role="user", content=req.prompt)], max_tokens=req.max_new_tokens, temperature=req.temperature)
    res = openai_chat_completions(chat_req, api_key=api_key)
    return {
        "question": req.prompt,
        "answer": res["choices"][0]["message"]["content"],
        "tokens": res["usage"]["completion_tokens"],
        "tok_per_sec": res["performance"]["tokens_per_sec"]
    }
