import math
import pytest
import torch
from tinylm.config import TinyLMConfig, get_device
from tinylm.model import TinyLM, CausalSelfAttention, LayerNorm


def test_layernorm_shape_and_stats():
    ln = LayerNorm(d_model=64)
    x = torch.randn(2, 10, 64)
    out = ln(x)
    assert out.shape == (2, 10, 64)
    # Output should have mean ~ 0 and std ~ 1 across last dim
    assert torch.allclose(out.mean(dim=-1), torch.zeros(2, 10), atol=1e-4)


def test_attention_causality():
    """
    CRITICAL TEST: Ensure causal self-attention does NOT leak future information.
    If we change a token in the future (position t=3), the outputs at positions
    t=0, 1, 2 must remain 100% IDENTICAL.
    """
    config = TinyLMConfig.toy()
    attn = CausalSelfAttention(config)
    attn.eval()  # Disable dropout for deterministic check

    # Input 1: Sequence of 4 vectors
    x1 = torch.randn(1, 4, config.d_model)
    # Input 2: Same first 3 vectors, but 4th vector is completely different!
    x2 = x1.clone()
    x2[:, 3, :] = torch.randn(1, config.d_model)

    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)

    # Outputs for tokens 0, 1, 2 must be identical
    assert torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-6), (
        "Causal mask failed! Past tokens were affected by future token changes."
    )
    # Output for token 3 should be different
    assert not torch.allclose(out1[:, 3, :], out2[:, 3, :])


def test_tinylm_forward_pass_and_loss():
    config = TinyLMConfig.toy()
    model = TinyLM(config)
    model.eval()

    B, T = 2, 8
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    logits, loss = model(idx, targets=targets)

    # Check shapes
    assert logits.shape == (B, T, config.vocab_size)
    assert loss is not None
    assert loss.dim() == 0  # Scalar loss
    assert loss.item() > 0.0

    # For randomly initialized weights, loss should be close to -ln(1/vocab_size) = ln(vocab_size)
    expected_loss = math.log(config.vocab_size)
    # Give a reasonable initial tolerance for random normal weights
    assert abs(loss.item() - expected_loss) < 2.0


def test_model_parameter_counts():
    toy = TinyLM(TinyLMConfig.toy())
    s = TinyLM(TinyLMConfig.tiny_s())
    m = TinyLM(TinyLMConfig.tiny_m())

    toy_params = toy.get_num_params()
    s_params = s.get_num_params()
    m_params = m.get_num_params()

    print(f"Toy params: {toy_params:,}")
    print(f"TinyLM-S params: {s_params:,}")
    print(f"TinyLM-M params: {m_params:,}")

    assert toy_params < 1_000_000
    # TinyLM-S should be roughly 5-15M params depending on vocab size
    assert 5_000_000 <= s_params <= 20_000_000
    # TinyLM-M should be larger
    assert m_params > s_params


def test_mps_device_forward():
    device = get_device()
    config = TinyLMConfig.toy()
    model = TinyLM(config).to(device)
    idx = torch.randint(0, config.vocab_size, (2, 4), device=device)

    logits, _ = model(idx)
    assert logits.device.type == device.type
    assert logits.shape == (2, 4, config.vocab_size)
