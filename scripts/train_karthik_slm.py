#!/usr/bin/env python3
"""
Train/Fine-Tune Karthik Jayan Portfolio SLM on Apple Silicon using Apple MLX.
Uses verified facts from karthikjayan.dev.
"""

import subprocess
from pathlib import Path

DATA_DIR = Path("data/mlx_karthik")
ADAPTER_PATH = Path("checkpoints/mlx_karthik_adapters_1.5b")
MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def generate_data():
    print("📋 Generating verified portfolio training pairs from karthikjayan.dev...")
    subprocess.run(["uv", "run", "python", "scripts/generate_verified_corpus.py"], check=True)


def train_model():
    print("=" * 60)
    print(f"🚀 Training Portfolio SLM with Apple MLX on {MODEL_ID}...")
    print("=" * 60)

    cmd = [
        "uv", "run", "mlx_lm.lora",
        "--model", MODEL_ID,
        "--data", str(DATA_DIR),
        "--train",
        "--mask-prompt",
        "--batch-size", "4",
        "--iters", "300",
        "--learning-rate", "1e-4",
        "--steps-per-report", "25",
        "--steps-per-eval", "50",
        "--adapter-path", str(ADAPTER_PATH),
        "--save-every", "100",
    ]

    subprocess.run(cmd, check=True)
    print("\n✅ Fine-tuning complete! Adapters saved to:", ADAPTER_PATH)


if __name__ == "__main__":
    generate_data()
    train_model()
