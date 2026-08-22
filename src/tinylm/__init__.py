"""
TinyLM: A decoder-only Small Language Model trained from scratch in PyTorch.
"""

from tinylm.config import TinyLMConfig, get_device
from tinylm.tokenizer import Tokenizer
from tinylm.data import TextDataset, create_dataloader, load_tinyshakespeare
from tinylm.model import TinyLM, CausalSelfAttention, TransformerBlock, LayerNorm, MLP
from tinylm.trainer import Trainer, TrainerConfig, configure_optimizers, get_lr
from tinylm.generate import sample_next_token, generate, generate_text
from tinylm.rope import precompute_rope_frequencies, apply_rope
from tinylm.eval import compute_perplexity, benchmark_throughput
from tinylm.server import run_server

__all__ = [
    "TinyLMConfig",
    "get_device",
    "Tokenizer",
    "TextDataset",
    "create_dataloader",
    "load_tinyshakespeare",
    "TinyLM",
    "CausalSelfAttention",
    "TransformerBlock",
    "LayerNorm",
    "MLP",
    "Trainer",
    "TrainerConfig",
    "configure_optimizers",
    "get_lr",
    "sample_next_token",
    "generate",
    "generate_text",
    "precompute_rope_frequencies",
    "apply_rope",
    "compute_perplexity",
    "benchmark_throughput",
    "run_server",
]
