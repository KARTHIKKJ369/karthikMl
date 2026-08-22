import math
import pytest
import torch
from tinylm.config import TinyLMConfig, get_device
from tinylm.model import TinyLM
from tinylm.data import create_dataloader
from tinylm.eval import compute_perplexity, benchmark_throughput


def test_compute_perplexity_math():
    config = TinyLMConfig.toy()
    model = TinyLM(config)

    tokens = torch.randint(0, config.vocab_size, (100,), dtype=torch.long)
    loader = create_dataloader(tokens, batch_size=2, context_length=8)

    loss, ppl = compute_perplexity(model, loader)

    assert loss > 0.0
    assert ppl > 1.0
    assert abs(ppl - math.exp(loss)) < 1e-4


def test_benchmark_throughput():
    device = get_device()
    config = TinyLMConfig.toy()
    model = TinyLM(config).to(device)

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)
    metrics = benchmark_throughput(model, prompt, num_tokens=10, device=device)

    assert "tokens_per_sec" in metrics
    assert "total_time_sec" in metrics
    assert metrics["tokens_generated"] == 10
    assert metrics["tokens_per_sec"] > 0.0
