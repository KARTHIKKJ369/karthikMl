import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from tinylm.config import TinyLMConfig
from tinylm.rope import precompute_rope_frequencies, apply_rope


class LayerNorm(nn.Module):
    """
    Layer Normalization with optional learnable bias.
    Normalizes activations across the last dimension (d_model) to have mean=0, std=1.
    """

    def __init__(self, d_model: int, bias: bool = False, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (gamma)
        self.weight = nn.Parameter(torch.ones(d_model))
        # Optional learnable shift parameter (beta)
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Sequence_Length, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # Scale and optional shift
        out = x_norm * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class CausalSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention with optional Rotary Positional Embeddings (RoPE).
    Computes Scaled Dot-Product Attention with a lower-triangular causal mask.
    """

    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout
        self.use_rope = config.use_rope

        # Fused linear projection for all Query, Key, and Value matrices
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)

        # Output projection to combine all attention heads back into d_model
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Precompute RoPE frequencies if enabled
        if self.use_rope:
            cos, sin = precompute_rope_frequencies(self.head_dim, config.context_length)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)

        # Register causal lower-triangular mask as a persistent buffer
        mask = torch.tril(torch.ones(config.context_length, config.context_length)).view(
            1, 1, config.context_length, config.context_length
        )
        self.register_buffer("bias_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Causal Self-Attention.
        
        Args:
            x: Input tensor of shape (Batch_Size, Sequence_Length, d_model)
        Returns:
            Output tensor of shape (Batch_Size, Sequence_Length, d_model)
        """
        B, T, C = x.shape  # Batch size, Sequence length (T <= context_length), Embedding dim (C = d_model)

        # Step 1: Project x into Q, K, V
        # Shape: (B, T, 3 * C)
        qkv = self.c_attn(x)
        # Split into separate Q, K, V each of shape (B, T, C)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Step 2: Split into multi-head representations and transpose:
        # Reshape: (B, T, n_heads, head_dim) -> Transpose: (B, n_heads, T, head_dim)
        # Transposing puts 'n_heads' in dimension 1 so batch matrix multiplication operates on (T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Step 3: Apply Rotary Positional Embeddings (RoPE) if enabled
        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Step 4: Compute raw attention scores: Q @ K^T / sqrt(head_dim)
        # k.transpose(-2, -1) shape: (B, n_heads, head_dim, T)
        # att shape: (B, n_heads, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Step 4: Apply Causal Mask
        # Future positions (where mask == 0) are filled with -infinity so softmax outputs 0.0
        att = att.masked_fill(self.bias_mask[:, :, :T, :T] == 0, float("-inf"))

        # Step 5: Softmax to get attention probability distribution over past tokens
        att_weights = F.softmax(att, dim=-1)
        att_weights = self.attn_dropout(att_weights)

        # Step 6: Multiply attention weights by values V
        # (B, n_heads, T, T) @ (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim)
        y = att_weights @ v

        # Step 7: Re-assemble all head outputs side-by-side (concatenate)
        # (B, n_heads, T, head_dim) -> Transpose: (B, T, n_heads, head_dim) -> Contiguous View: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Step 8: Final linear output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Expands d_model -> d_ff (4 * d_model), applies GELU non-linearity, and projects back -> d_model.
    """

    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, d_model) -> (B, T, d_ff)
        x = self.c_fc(x)
        x = self.gelu(x)
        # (B, T, d_ff) -> (B, T, d_model)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer Decoder Block with Pre-LayerNorm and Residual Connections.
    
    x = x + Attention(LayerNorm1(x))
    x = x + MLP(LayerNorm2(x))
    """

    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm 1 + Attention + Residual Add
        x = x + self.attn(self.ln_1(x))
        # Pre-norm 2 + MLP + Residual Add
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyLM(nn.Module):
    """
    TinyLM: Decoder-Only Small Language Model.
    """

    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        self.config = config

        # Token Embedding: maps token integer IDs [0..vocab_size-1] -> d_model vectors
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        # Positional Embedding: learned vectors for positions [0..context_length-1]
        self.position_embeddings = nn.Embedding(config.context_length, config.d_model)

        self.drop = nn.Dropout(config.dropout)

        # Stack of N Transformer Blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Final LayerNorm
        self.ln_f = LayerNorm(config.d_model, bias=config.bias)

        # Language Model Head: maps final representations (d_model) -> vocabulary logits (vocab_size)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying (Weight Sharing):
        # Tie the weights of token embeddings and lm_head (standard in GPT models)
        self.token_embeddings.weight = self.lm_head.weight

        # Initialize all model weights with standard normal distribution (std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights with standard normal distribution (std=0.02)
        similar to GPT-2 initialization.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Return the total number of trainable parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= self.position_embeddings.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for TinyLM.
        
        Args:
            idx: Input token IDs tensor of shape (Batch_Size, Sequence_Length)
            targets: Optional ground-truth target token IDs of shape (Batch_Size, Sequence_Length)
            
        Returns:
            logits: Output tensor of shape (Batch_Size, Sequence_Length, vocab_size)
            loss: Cross-entropy scalar loss (if targets provided), else None
        """
        device = idx.device
        B, T = idx.shape

        if T > self.config.context_length:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max context length is {self.config.context_length}"
            )

        # Embeddings
        tok_emb = self.token_embeddings(idx)           # Shape: (B, T, d_model)
        if self.config.use_rope:
            # RoPE handles positions dynamically inside attention layers
            x = self.drop(tok_emb)
        else:
            # Learned absolute positional embeddings
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            pos_emb = self.position_embeddings(pos)        # Shape: (T, d_model) broadcast to (B, T, d_model)
            x = self.drop(tok_emb + pos_emb)

        # Pass through all Transformer Blocks sequentially
        for block in self.blocks:
            x = block(x)

        # Final LayerNorm
        x = self.ln_f(x)                               # Shape: (B, T, d_model)

        # Compute logits over vocabulary
        logits = self.lm_head(x)                       # Shape: (B, T, vocab_size)

        # Compute Cross-Entropy Loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten B and T dimensions to compute cross-entropy across all tokens in batch:
            # logits: (B * T, vocab_size), targets: (B * T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
