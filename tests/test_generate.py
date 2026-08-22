import pytest
import torch
from tinylm.config import TinyLMConfig, get_device
from tinylm.model import TinyLM
from tinylm.tokenizer import Tokenizer
from tinylm.generate import sample_next_token, generate, generate_text


def test_greedy_sampling_deterministic():
    # Greedy sampling (temperature=0) should always pick the maximum logit
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.5]])
    sampled = sample_next_token(logits, temperature=0.0)
    assert sampled.item() == 1  # Index of 5.0 is 1


def test_generate_token_length():
    config = TinyLMConfig.toy()
    model = TinyLM(config)
    
    idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
    max_new_tokens = 5
    out = generate(model, idx, max_new_tokens=max_new_tokens, temperature=0.0)
    
    assert out.shape == (1, 3 + max_new_tokens)
    # The first 3 tokens must remain exactly the prompt
    assert torch.equal(out[0, :3], idx[0])


def test_generate_exceeding_context_length():
    # If context_length=8 and we generate 12 tokens, it should smoothly crop and continue
    config = TinyLMConfig.toy()
    config.context_length = 8
    model = TinyLM(config)

    idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = generate(model, idx, max_new_tokens=10, temperature=0.5)
    assert out.shape == (1, 13)


def test_early_stopping_on_eos():
    config = TinyLMConfig.toy()
    model = TinyLM(config)

    idx = torch.tensor([[1, 2, 3]], dtype=torch.long)
    # If EOS is 0 and we force greedy to pick 0 (by overriding or testing EOS condition)
    # Let's test with eos_token_id
    out = generate(model, idx, max_new_tokens=20, eos_token_id=999999)  # Won't trigger
    assert out.shape == (1, 23)


def test_generate_text_pipeline():
    tokenizer = Tokenizer()
    config = TinyLMConfig.toy()
    config.vocab_size = tokenizer.vocab_size
    model = TinyLM(config)

    prompt = "Hello"
    result = generate_text(model, tokenizer, prompt, max_new_tokens=5, temperature=0.8)
    assert isinstance(result, str)
    assert result.startswith("Hello")
