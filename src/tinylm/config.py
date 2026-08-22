from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class TinyLMConfig:
    """
    Configuration blueprint for TinyLM model architecture.
    """
    # Vocabulary & Context
    vocab_size: int = 50257        # Number of unique tokens (default: GPT-2 tokenizer vocab size)
    context_length: int = 256      # Maximum sequence length (block size / context window)

    # Transformer Dimensions
    d_model: int = 256             # Embedding dimension (width of the model)
    n_layers: int = 6              # Number of Transformer blocks stacked vertically
    n_heads: int = 8               # Number of attention heads
    d_ff: Optional[int] = None     # Hidden dimension of MLP/Feed-Forward network (defaults to 4 * d_model)

    # Regularization & Architectural Details
    dropout: float = 0.1           # Dropout probability (0.0 to disable during inference)
    bias: bool = False             # Whether to use bias in Linear layers and LayerNorm (False is standard in modern LLMs)
    use_rope: bool = False         # Whether to use Rotary Positional Embeddings (enabled in Milestone 5)

    def __post_init__(self) -> None:
        """
        Validate hyperparameters and derive dependent dimensions.
        """
        # Ensure d_model can be evenly split across attention heads
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

        # Standard transformer FFN expands dimension by 4x if not explicitly specified
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

    @property
    def head_dim(self) -> int:
        """
        Dimension of each individual attention head: d_k = d_model / n_heads.
        """
        return self.d_model // self.n_heads

    @classmethod
    def toy(cls) -> "TinyLMConfig":
        """
        Ultra-lightweight preset for fast debugging and unit tests.
        """
        return cls(
            vocab_size=1000,
            context_length=64,
            d_model=64,
            n_layers=2,
            n_heads=2,
            dropout=0.0,
        )

    @classmethod
    def tiny_s(cls) -> "TinyLMConfig":
        """
        TinyLM-S preset: ~5M parameters. Fast to train on Apple Silicon M4.
        """
        return cls(
            vocab_size=50257,
            context_length=256,
            d_model=256,
            n_layers=6,
            n_heads=8,
            dropout=0.1,
        )

    @classmethod
    def tiny_m(cls) -> "TinyLMConfig":
        """
        TinyLM-M preset: ~20M parameters.
        """
        return cls(
            vocab_size=50257,
            context_length=512,
            d_model=512,
            n_layers=8,
            n_heads=8,
            dropout=0.1,
        )


def get_device() -> torch.device:
    """
    Automatically detect best available hardware:
    1. 'mps' for Apple Silicon GPU (M-series)
    2. 'cuda' for NVIDIA GPU
    3. 'cpu' fallback
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
