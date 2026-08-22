import math
import pytest
import torch
from tinylm.config import TinyLMConfig, get_device
from tinylm.model import TinyLM, CausalSelfAttention
from tinylm.rope import precompute_rope_frequencies, apply_rope


def test_rope_norm_preservation():
    """
    Rotation in 2D Euclidean space preserves vector length (norm).
    ||RoPE(x)|| == ||x||
    """
    dim = 32
    seq_len = 16
    cos, sin = precompute_rope_frequencies(dim, max_seq_len=seq_len)

    # Input tensor of shape (B=1, n_heads=1, T=seq_len, head_dim=dim)
    x = torch.randn(1, 1, seq_len, dim)
    x_rot = apply_rope(x, cos, sin)

    norm_before = torch.norm(x, dim=-1)
    norm_after = torch.norm(x_rot, dim=-1)

    assert torch.allclose(norm_before, norm_after, atol=1e-5), (
        "RoPE failed to preserve vector norms across sequence positions!"
    )


def test_rope_relative_position_invariance():
    """
    MATHEMATICAL FOUNDATION OF RoPE:
    The dot product between query at position (m + delta) and key at position (n + delta)
    depends ONLY on the distance (m - n), so shifting both by delta leaves the dot product unchanged!
    
    <RoPE(q, m), RoPE(k, n)> == <RoPE(q, m + delta), RoPE(k, n + delta)>
    """
    dim = 32
    max_len = 64
    cos, sin = precompute_rope_frequencies(dim, max_seq_len=max_len)

    # Pick two arbitrary token vectors
    q_vec = torch.randn(1, 1, 1, dim)
    k_vec = torch.randn(1, 1, 1, dim)

    # Position m=5 and position n=2 (relative distance = 3)
    m, n = 5, 2
    delta = 10  # Shift both positions by 10 -> (15, 12)

    # Rotate at (5, 2)
    q_5 = apply_rope(q_vec, cos[m:m+1], sin[m:m+1])
    k_2 = apply_rope(k_vec, cos[n:n+1], sin[n:n+1])
    dot_product_1 = (q_5 * k_2).sum()

    # Rotate at shifted positions (15, 12)
    q_15 = apply_rope(q_vec, cos[m+delta : m+delta+1], sin[m+delta : m+delta+1])
    k_12 = apply_rope(k_vec, cos[n+delta : n+delta+1], sin[n+delta : n+delta+1])
    dot_product_2 = (q_15 * k_12).sum()

    assert torch.allclose(dot_product_1, dot_product_2, atol=1e-5), (
        f"RoPE failed relative distance invariance! Dot1: {dot_product_1.item()}, Dot2: {dot_product_2.item()}"
    )


def test_tinylm_with_rope_forward_pass():
    config = TinyLMConfig.toy()
    config.use_rope = True
    model = TinyLM(config)

    idx = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    logits, loss = model(idx, targets=targets)

    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is not None
    assert loss.item() > 0.0


def test_rope_attention_causality():
    config = TinyLMConfig.toy()
    config.use_rope = True
    attn = CausalSelfAttention(config)
    attn.eval()

    x1 = torch.randn(1, 4, config.d_model)
    x2 = x1.clone()
    x2[:, 3, :] = torch.randn(1, config.d_model)

    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)

    assert torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-6)
    assert not torch.allclose(out1[:, 3, :], out2[:, 3, :])
