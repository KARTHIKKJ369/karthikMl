# ⚡ TinyLM & Portfolio AI Engine

> **A dual-architecture AI project**: A decoder-only Small Language Model built from scratch in pure **PyTorch** and an **Apple MLX fine-tuned Portfolio Intelligence SLM (Qwen2.5-1.5B)** served via an **OpenAI-compatible Headless API** for seamless deployment on Azure VMs.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Apple MLX](https://img.shields.io/badge/Apple_MLX-Metal_MPS-000000?style=flat&logo=apple&logoColor=white)](https://github.com/ml-explore/mlx)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Architectural Features

### 1. Pure PyTorch TinyLM (Built From Scratch)
- **Decoder-Only Transformer**: Autoregressive next-token prediction with causal upper-triangular masking.
- **Rotary Positional Embeddings (RoPE)**: 2D rotation of Query ($Q$) and Key ($K$) vectors preserving relative token distances without static context length caps.
- **Pre-Layer Normalization**: Stabilizes gradient flow across deep transformer layers ($x = x + \text{Sublayer}(\text{LN}(x))$).
- **Decoupled AdamW Optimizer**: Weight decay ($0.1$) is strictly applied to 2D weight matrices, while 1D normalization scales ($\gamma$) and biases ($\beta$) receive $0.0$ decay.
- **Cosine Learning Rate Scheduling**: Linear warmup from $0 \rightarrow \text{lr}_{\text{max}}$ followed by cosine decay down to $10\%$ peak LR.
- **Autoregressive Sampling Engine**: Temperature scaling, Top-$k$ filtering, Top-$p$ (nucleus) sampling, and repetition penalty.
- **Test Suite**: 33/33 comprehensive unit tests passing with 100% success rate.

### 2. Karthik Jayan Portfolio SLM (Apple MLX Fine-Tuned)
- **Base Architecture**: `Qwen2.5-1.5B-Instruct-4bit` fine-tuned via Apple MLX (`mlx-lm`) using LoRA adapters on Apple Silicon GPU (`mps`/Metal).
- **Memory Footprint**: Only **~1.1 GB RAM** at runtime.
- **Inference Speed**: **60–80+ tokens/sec** on Apple Silicon M4.
- **Verified Portfolio Grounding**: Fine-tuned on verified engineering facts extracted directly from [`karthikjayan.dev`](https://karthikjayan.dev):
  - **Ridge CRAG Platform**: Multi-hop Self-Correcting RAG with LangGraph, Groq, ChromaDB + BM25 (RRF $K=60$), FlashRank, and Azure VM hosting (`ridge.karthikjayan.tech`).
  - **Recall Memory**: Persistent LLM memory architecture with decay modeling and semantic caching (Accepted at **JETIR**).
  - **CyberLabs IIIT Kottayam**: Backwater Collision Avoidance with ESP32, LoRa telemetry, MQTT, and LSTM trajectory prediction.
  - **Academic Background**: B.Tech CSE ('27) at Mar Athanasius College of Engineering (MACE, Autonomous) with a **9.14 CGPA**.

### 3. Production Headless API & Azure Deployment
- **API Key Security**: Header authentication (`x-api-key: <KEY>` or `Authorization: Bearer <KEY>`).
- **OpenAI SDK Compatible**: Implements `POST /v1/chat/completions` for drop-in integration with the official `openai` npm/pip package in portfolio frontends.
- **Direct REST Endpoint**: Lightweight `POST /api/chat` and `POST /api/generate`.
- **CORS Configured**: Pre-configured for `https://karthikjayan.dev` and `https://ridge.karthikjayan.tech`.
- **Docker & Compose**: 1-command deployment to Microsoft Azure Linux VMs.

### 4. Anti-AI-Slop Ridge Web Playground
- Editorial, typography-first interface inspired by [`ridge.karthikjayan.tech`](https://ridge.karthikjayan.tech).
- **3 Color Themes**: `Void` (Obsidian Dark), `Stone` (Editorial Newsprint), and `Rust` (Terracotta Charcoal).
- **Typography**: *Newsreader* serif headings, *Plus Jakarta Sans* body, and *JetBrains Mono* telemetry metrics.

---

## 📐 Architecture Diagram

```
[User / Recruiter on karthikjayan.dev]
                │
                ▼ (HTTPS / Cloudflare Tunnel / Azure VM)
┌─────────────────────────────────────────────────────────────┐
│  Headless API Server (FastAPI + Uvicorn)                   │
│    ├── API Key Auth (x-api-key / Bearer Token)              │
│    ├── CORS Middleware (karthikjayan.dev)                   │
│    ├── POST /v1/chat/completions (OpenAI Compatible)        │
│    └── POST /api/chat                                       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Fine-Tuned Portfolio SLM (1.5B 4-bit Quantized)            │
│    ├── Base: Qwen2.5-1.5B-Instruct-4bit                     │
│    ├── LoRA Adapters: checkpoints/mlx_karthik_adapters_1.5b │
│    ├── System Persona Grounding (Ridge, Recall, CyberLabs)  │
│    └── Metal GPU / x86 Engine (~1.1 GB RAM Footprint)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Installation & Environment Setup
Manage dependencies effortlessly with `uv`:

```bash
# Clone repository
git clone https://github.com/KARTHIKKJ369/karthikMl.git
cd karthikMl

# Install dependencies and editable package
uv pip install -e .
```

### 2. Run the Test Suite
Verify model logic, RoPE invariants, trainer optimizers, and generation kernels:

```bash
uv run pytest
```
*Result: 33 passed in ~5.4s.*

---

## 🔌 API Integration Guide

### Call the API from Your Portfolio (`karthikjayan.dev`)

#### Option A: Direct TypeScript / JavaScript `fetch`
```typescript
async function askKarthikAI(question: string) {
  const response = await fetch("https://api.karthikjayan.dev/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": "Bearer kj_live_sec_789f2a4b1c", // your API key
    },
    body: JSON.stringify({
      model: "karthik-qwen2.5-1.5b",
      messages: [{ role: "user", content: question }],
      temperature: 0.1,
      max_tokens: 250,
    }),
  });

  const data = await response.json();
  return data.choices[0].message.content;
}
```

#### Option B: Official OpenAI SDK (`npm install openai`)
```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://api.karthikjayan.dev/v1",
  apiKey: "kj_live_sec_789f2a4b1c",
  dangerouslyAllowBrowser: true,
});

const res = await client.chat.completions.create({
  model: "karthik-qwen2.5-1.5b",
  messages: [{ role: "user", content: "Explain your Ridge RAG project." }],
});

console.log(res.choices[0].message.content);
```

---

## ☁️ Azure VM Deployment (1-Command Docker)

Deploy to any standard Microsoft Azure Linux VM:

```bash
# 1. Clone repository on your Azure VM
git clone https://github.com/KARTHIKKJ369/karthikMl.git
cd karthikMl

# 2. Launch with Docker Compose
docker compose up -d
```

Or run directly with Python / Uvicorn:
```bash
export PORTFOLIO_API_KEY="your-custom-secret-key"
uv run uvicorn src.tinylm.api:app --host 0.0.0.0 --port 8000
```

---

## 🏋️ Fine-Tuning & Dataset Generation

To re-tune or update the portfolio knowledge base:

```bash
# 1. Generate multi-query verified dataset from karthikjayan.dev
uv run python scripts/generate_verified_corpus.py

# 2. Fine-tune with Apple MLX LoRA
uv run mlx_lm.lora \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --data data/mlx_karthik \
  --train \
  --mask-prompt \
  --batch-size 4 \
  --iters 300 \
  --learning-rate 1e-4 \
  --adapter-path checkpoints/mlx_karthik_adapters_1.5b
```

---

## 📁 Repository Structure

```
.
├── pyproject.toml                     # uv configuration and dependencies
├── Dockerfile                         # Production Docker container for Azure VM
├── docker-compose.yml                 # 1-command Docker deployment
├── README.md                          # Full technical documentation
├── data/
│   └── mlx_karthik/                   # Generated train and valid JSONL pairs
├── checkpoints/
│   └── mlx_karthik_adapters_1.5b/     # Fine-tuned Apple MLX LoRA adapters
├── src/
│   └── tinylm/
│       ├── __init__.py                # Package exports
│       ├── api.py                     # Headless FastAPI server (OpenAI compatible + API Auth)
│       ├── config.py                  # TinyLMConfig dataclass & presets
│       ├── tokenizer.py               # GPT-2 BPE Tokenizer wrapper
│       ├── data.py                    # Dataset, DataLoader, & TinyShakespeare
│       ├── model.py                   # Decoder-only Transformer & Pre-LN Blocks
│       ├── rope.py                    # Rotary Positional Embeddings (RoPE)
│       ├── trainer.py                 # AdamW optimizer & Cosine LR Scheduler
│       ├── generate.py                # Sampling engine (Top-k, Top-p, Temperature)
│       ├── eval.py                    # Perplexity & tok/s speed benchmarking
│       ├── server.py                  # Playground HTTP API backend
│       └── web/                       # Ridge-styled web playground frontend
│           ├── index.html
│           ├── style.css
│           └── app.js
├── scripts/
│   ├── generate_verified_corpus.py    # Generates fine-tuning corpus from karthikjayan.dev
│   ├── export_fused_model.py          # Fuses LoRA weights for standalone deployment
│   ├── train_tinylm_s.py              # Pre-training script for TinyLM-S
│   ├── train_tinylm_m.py              # Pre-training script for TinyLM-M
│   ├── evaluate_models.py             # Evaluation & scaling benchmark script
│   └── serve_playground.py            # Local playground server
└── tests/                             # Full unit test suite (33/33 tests passing)
```

---

## 📜 License
MIT License. Built for production AI systems, LLM fine-tuning, and modern portfolio intelligence.
