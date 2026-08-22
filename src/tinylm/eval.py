import math
import time
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tinylm.config import get_device
from tinylm.model import TinyLM
from tinylm.generate import generate


@torch.no_grad()
def compute_perplexity(
    model: TinyLM,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """
    Compute average Cross-Entropy loss and Perplexity over an entire dataset / validation split.
    
    Returns:
        (avg_loss, perplexity)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    total_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        total_loss += loss.item()
        total_batches += 1

    if total_batches == 0:
        return 0.0, 1.0

    avg_loss = total_loss / total_batches
    perplexity = math.exp(min(avg_loss, 100.0))  # Clamp to prevent overflow
    return avg_loss, perplexity


@torch.no_grad()
def benchmark_throughput(
    model: TinyLM,
    prompt_tokens: torch.Tensor,
    num_tokens: int = 100,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Benchmark autoregressive generation speed (tokens/sec) and latency.
    
    Args:
        model: TinyLM instance
        prompt_tokens: Tensor of shape (1, Sequence_Length)
        num_tokens: Number of tokens to generate
        device: Target compute device
        
    Returns:
        Dict with 'tokens_per_sec', 'total_time', and 'tokens_generated'
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    prompt_tokens = prompt_tokens.to(device)

    # Warmup pass (important for GPU / MPS cache and kernel compilation)
    _ = generate(model, prompt_tokens, max_new_tokens=5, temperature=0.0)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    # Timed generation
    start_time = time.perf_counter()
    out = generate(model, prompt_tokens, max_new_tokens=num_tokens, temperature=0.8)
    
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    tokens_generated = out.shape[1] - prompt_tokens.shape[1]
    tokens_per_sec = tokens_generated / max(elapsed, 1e-6)

    return {
        "tokens_per_sec": tokens_per_sec,
        "total_time_sec": elapsed,
        "tokens_generated": tokens_generated,
    }
