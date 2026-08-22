from typing import Tuple, Optional
import torch


def precompute_rope_frequencies(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cosine and sine frequency tensors for Rotary Positional Embeddings (RoPE).
    
    Args:
        dim: Dimension of each attention head (head_dim). Must be even.
        max_seq_len: Maximum sequence length (context_length).
        theta: Base frequency constant (default: 10000.0 from LLaMA/RoFormer).
        device: Device to allocate tensors on.
        
    Returns:
        cos: Cosine tensor of shape (max_seq_len, dim)
        sin: Sine tensor of shape (max_seq_len, dim)
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension ({dim}) must be even.")

    # 1. Compute angular frequencies for each pair of dimensions:
    # theta_i = 1.0 / (theta ** (2i / dim)) for i in [0, 1, ..., dim/2 - 1]
    # Shape: (dim / 2,)
    indices = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    freqs = 1.0 / (theta ** (indices / dim))

    # 2. Compute position indices: [0, 1, 2, ..., max_seq_len - 1]
    # Shape: (max_seq_len,)
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)

    # 3. Outer product between position indices and frequencies:
    # Shape: (max_seq_len, dim / 2)
    freqs_table = torch.outer(t, freqs)

    # 4. Duplicate frequencies across the two paired dimensions so shape matches full `dim`:
    # Shape: (max_seq_len, dim)
    freqs_table = torch.cat((freqs_table, freqs_table), dim=-1)

    cos = torch.cos(freqs_table)
    sin = torch.sin(freqs_table)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the second half of dimensions into the first with opposite signs:
    For x = [x1, x2], returns [-x2, x1].
    """
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply Rotary Positional Embedding to input Query or Key tensor x.
    
    Args:
        x: Tensor of shape (Batch_Size, n_heads, Sequence_Length, head_dim)
        cos: Cosine tensor of shape (Sequence_Length, head_dim)
        sin: Sine tensor of shape (Sequence_Length, head_dim)
        
    Returns:
        Rotated tensor with same shape as x.
    """
    T = x.shape[2]
    # Slice cos and sin up to current sequence length T
    # Reshape from (T, head_dim) to (1, 1, T, head_dim) for broadcasting over (B, n_heads, T, head_dim)
    cos = cos[:T, :].unsqueeze(0).unsqueeze(0).to(x.device, dtype=x.dtype)
    sin = sin[:T, :].unsqueeze(0).unsqueeze(0).to(x.device, dtype=x.dtype)

    # Fast 2D rotation formula: x * cos + rotate_half(x) * sin
    # [x1, x2] * cos + [-x2, x1] * sin = [x1*cos - x2*sin, x2*cos + x1*sin]
    return (x * cos) + (rotate_half(x) * sin)
