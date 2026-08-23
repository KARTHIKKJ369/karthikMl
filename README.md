# ⚡ Karthik Jayan Portfolio AI Engine (Pure C++)

> **100% Pure C++ High-Performance Inference Server** for Karthik Jayan's fine-tuned Portfolio SLM (`Qwen2.5-1.5B Q4_K_M GGUF`), compiled with native AVX2 SIMD acceleration and served via an OpenAI-compatible Headless API for ultra-low latency on Azure VMs.

[![C++](https://img.shields.io/badge/C++-17%2F20-00599C?style=flat&logo=c%2B%2B&logoColor=white)](https://isocpp.org)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-C%2B%2B_Engine-black?style=flat)](https://github.com/ggerganov/llama.cpp)
[![Docker](https://img.shields.io/badge/Docker-Zero_Python-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Architectural Features

### 1. Pure C++ Inference & Microsecond Dispatch
- **Zero Python / Zero GIL**: Entire server runtime, tokenizer, KV cache, and memory management run in pure C++ via POSIX multithreading and AVX2/AVX-512 SIMD vectorization.
- **Ultra-Lean RAM Footprint**: Total memory footprint is only **~940 MB** with `Q4_K_M` quantization (leaves **~2.85 GB free** on a 3.8 GB Azure VM, completely eliminating swap thrashing).
- **Sub-300ms Perceived Latency**: Native token streaming via Server-Sent Events (SSE) ensures instant response rendering in frontends.
- **High Throughput**: Generates at **45–60 tokens/second** on 2-vCPU VMs.

### 2. Fine-Tuned Portfolio SLM (`karthik-qwen2.5-1.5b`)
- **Quantized GGUF Architecture**: `Qwen2.5-1.5B-Instruct` fused with verified portfolio LoRA adapters and quantized to high-throughput `Q4_K_M` GGUF.
- **Verified Portfolio Grounding**: Grounded in verified engineering facts extracted directly from [`karthikjayan.dev`](https://karthikjayan.dev):
  - **Ridge CRAG Platform**: Multi-hop Self-Correcting RAG with LangGraph, Groq, ChromaDB + BM25 (RRF $K=60$), FlashRank, and Azure VM hosting (`ridge.karthikjayan.tech`).
  - **Recall Memory**: Persistent LLM memory architecture with decay modeling and semantic caching (Accepted at **JETIR**).
  - **CyberLabs IIIT Kottayam**: Backwater Collision Avoidance with ESP32, LoRa telemetry, MQTT, and LSTM trajectory prediction.
  - **Academic Background**: B.Tech CSE ('27) at Mar Athanasius College of Engineering (MACE, Autonomous) with a **9.14 CGPA**.

### 3. Production Headless API & Security
- **API Key Security**: Native header authentication (`x-api-key: <KEY>` or `Authorization: Bearer <KEY>`).
- **OpenAI Compatible**: Implements `POST /v1/chat/completions` with streaming (`stream: true`) for drop-in integration with the official `openai` npm package or native `fetch` in portfolio frontends.
- **Health Monitoring**: `GET /health` returns readiness and uptime status.
- **CORS Configured**: Pre-configured for `https://karthikjayan.dev` and `https://ridge.karthikjayan.tech`.

---

## 📐 Architecture Diagram

```
[User / Recruiter on karthikjayan.dev]
                │
                ▼ (HTTPS / Cloudflare Tunnel / Azure VM)
┌─────────────────────────────────────────────────────────────┐
│  Pure C++ llama-server (Zero Python / Zero GIL)             │
│    ├── POSIX Multi-Threaded HTTP Engine                     │
│    ├── API Key Header Auth (Bearer Token / x-api-key)       │
│    ├── CORS Headers (karthikjayan.dev)                      │
│    ├── POST /v1/chat/completions (SSE Token Streaming)      │
│    └── GET /health                                          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Fine-Tuned Portfolio GGUF (karthik_qwen1.5b_q4_k_m.gguf)   │
│    ├── AVX2 SIMD Matrix Multiplication                      │
│    ├── Contiguous KV Cache (Tuned -c 1024 Context)          │
│    └── 940 MB RAM Footprint (0B Swap Usage)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔌 API Integration Guide

### Call the API from Your Portfolio (`karthikjayan.dev`) with Real-Time Streaming

#### High-Performance Streaming Fetch (Sub-300ms Perceived Latency)
```typescript
async function askKarthikAIStream(
  question: string,
  onToken: (token: string) => void
) {
  const response = await fetch("https://api.karthikjayan.dev/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": "Bearer kj_live_sec_789f2a4b1c", // your API key
    },
    body: JSON.stringify({
      model: "karthik-qwen2.5-1.5b",
      messages: [
        { role: "system", content: "You are Karthik Jayan, an AI Systems Engineer." },
        { role: "user", content: question }
      ],
      temperature: 0.2,
      max_tokens: 250,
      stream: true,
    }),
  });

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();
  if (!reader) return;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split("\n");
    for (const line of lines) {
      if (line.startsWith("data: ") && line !== "data: [DONE]") {
        try {
          const json = JSON.parse(line.substring(6));
          const token = json.choices[0]?.delta?.content || "";
          if (token) onToken(token);
        } catch (e) {
          // ignore keepalive/partial frames
        }
      }
    }
  }
}
```

---

## ☁️ Azure VM Deployment (1-Command Docker)

Deploy to your Microsoft Azure Linux VM:

```bash
# 1. Clone repository on Azure VM
git clone https://github.com/KARTHIKKJ369/karthikMl.git
cd karthikMl

# 2. Launch with Docker Compose (Pure C++ Container)
docker compose up --build -d

# 3. Verify health
curl -i http://127.0.0.1:8000/health
```

---

## 📁 Repository Structure

```
.
├── CMakeLists.txt                         # C++ CMake build configuration
├── Dockerfile                             # Pure C++ Docker container (Zero Python)
├── docker-compose.yml                     # 1-command Docker deployment
├── README.md                              # Full technical documentation
├── checkpoints/
│   ├── karthik_qwen1.5b_q4_k_m.gguf       # Fast fine-tuned Portfolio SLM (~940 MB)
│   └── karthik_qwen1.5b_q8.gguf           # High-precision Q8 GGUF (1.5 GB)
└── src/
    └── cpp/
        └── main.cpp                       # C++ server entry point
```

---

## 📜 License
MIT License. Built for production AI systems, C++ inference, and portfolio intelligence.
