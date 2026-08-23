#!/usr/bin/env python3
"""
Deploy Karthik Jayan Portfolio SLM to 100% Free Hugging Face Spaces (Gradio SDK).
"""

from huggingface_hub import HfApi

REPO_ID = "karthikeee/karthik-portfolio-ai"
MODEL_LOCAL = "checkpoints/karthik_qwen1.5b_q4_k_m.gguf"

print(f"🚀 Initializing Free Gradio Space on Hugging Face: {REPO_ID}...")
api = HfApi()

# 1. Create Gradio Space (100% Free tier)
space_url = api.create_repo(
    repo_id=REPO_ID,
    repo_type="space",
    space_sdk="gradio",
    space_hardware="cpu-basic",
    exist_ok=True
)
print(f"✅ Space created: {space_url}")

# 2. Upload Space application files
print("📦 Uploading README.md, app.py, and requirements.txt...")
api.upload_file(
    path_or_fileobj="hf_space/README.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="space"
)
api.upload_file(
    path_or_fileobj="hf_space/app.py",
    path_in_repo="app.py",
    repo_id=REPO_ID,
    repo_type="space"
)
api.upload_file(
    path_or_fileobj="hf_space/requirements.txt",
    path_in_repo="requirements.txt",
    repo_id=REPO_ID,
    repo_type="space"
)

# 3. Upload GGUF model via Git LFS
print(f"⚡ Uploading GGUF model '{MODEL_LOCAL}' (~940 MB) to Hugging Face...")
api.upload_file(
    path_or_fileobj=MODEL_LOCAL,
    path_in_repo="karthik_qwen1.5b_q4_k_m.gguf",
    repo_id=REPO_ID,
    repo_type="space"
)

print("\n🎉 Deployment completed successfully!")
print(f"🌐 Public Space URL: https://huggingface.co/spaces/{REPO_ID}")
print(f"🔌 Public API Endpoint: https://{REPO_ID.replace('/', '-')}.hf.space")
