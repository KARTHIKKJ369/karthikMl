"""
Export / Merge Apple MLX LoRA adapters into standalone Hugging Face weights and GGUF.
Allows deploying on any standard Linux Azure VM (CPU or GPU).
"""

import argparse
import subprocess
from pathlib import Path
from mlx_lm import load
import mlx.core as mx

def export_fused_model(
    model_id: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    adapter_path: str = "checkpoints/mlx_karthik_adapters_1.5b",
    output_dir: str = "checkpoints/karthik_qwen1.5b_fused"
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Merging LoRA adapter '{adapter_path}' with base '{model_id}'...")
    
    cmd = [
        "uv", "run", "mlx_lm.fuse",
        "--model", model_id,
        "--adapter-path", adapter_path,
        "--save-path", output_dir
    ]
    subprocess.run(cmd, check=True)
    print(f"✅ Successfully exported fused model to '{output_dir}'.")

if __name__ == "__main__":
    export_fused_model()
