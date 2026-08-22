from typing import Optional, List, Union
import torch
import torch.nn.functional as F

from tinylm.config import TinyLMConfig, get_device
from tinylm.model import TinyLM
from tinylm.tokenizer import Tokenizer


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    generated_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sample next token from raw vocabulary logits using temperature, top-k, top-p, and repetition penalty.
    
    Args:
        logits: Tensor of shape (Batch_Size, Vocab_Size) representing next-token logits.
        temperature: Softmax temperature (> 0.0). Higher = more random, Lower = more deterministic.
        top_k: If set, retain only top k tokens with highest logits.
        top_p: If set, retain top tokens whose cumulative probability <= top_p (nucleus sampling).
        repetition_penalty: Multiplier penalty (> 1.0) applied to logits of tokens that already appeared.
        generated_tokens: Tensor of shape (Batch_Size, Sequence_Length) containing tokens seen so far.
        
    Returns:
        Sampled token IDs of shape (Batch_Size, 1).
    """
    # 1. Greedy decoding branch (temperature <= 0 or 1e-6)
    if temperature <= 1e-6:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 2. Apply repetition penalty: discount logits of tokens already present
    if repetition_penalty > 1.0 and generated_tokens is not None:
        for b in range(logits.shape[0]):
            unique_tokens = torch.unique(generated_tokens[b])
            for token_id in unique_tokens:
                # If logit is positive, divide by penalty; if negative, multiply by penalty
                if logits[b, token_id] > 0:
                    logits[b, token_id] /= repetition_penalty
                else:
                    logits[b, token_id] *= repetition_penalty

    # 3. Scale logits by temperature
    logits = logits / temperature

    # 4. Apply Top-K filtering
    if top_k is not None and top_k > 0:
        # Keep only the top k values; set everything below the k-th value to -inf
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1, None]  # The k-th highest logit value
        logits = torch.where(logits < min_values, torch.tensor(float("-inf"), device=logits.device), logits)

    # 5. Apply Top-P (Nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Shift cumulative probabilities to the right to keep the first token that exceeds top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter mask back to original logits positions
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # 6. Softmax to get valid probability distribution
    probs = F.softmax(logits, dim=-1)

    # 7. Sample 1 token from categorical distribution
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


@torch.no_grad()
def generate(
    model: TinyLM,
    idx: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.1,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Autoregressively generate new tokens given a conditioning prompt tensor `idx`.
    
    Args:
        model: Trained TinyLM instance.
        idx: Initial prompt token IDs of shape (Batch_Size, Sequence_Length).
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling threshold.
        top_p: Top-p (nucleus) threshold.
        repetition_penalty: Repetition penalty factor.
        eos_token_id: Optional token ID to stop early on (e.g. 50256 for <|endoftext|>).
        
    Returns:
        Tensor of shape (Batch_Size, Sequence_Length + generated_tokens)
    """
    model.eval()
    context_length = model.config.context_length

    for _ in range(max_new_tokens):
        # Crop context if sequence length exceeds model context length
        idx_cond = idx if idx.size(1) <= context_length else idx[:, -context_length:]

        # Forward model to get logits: (Batch_Size, T, Vocab_Size)
        logits, _ = model(idx_cond)

        # Extract logits at the last position only: (Batch_Size, Vocab_Size)
        next_token_logits = logits[:, -1, :]

        # Sample next token
        next_token = sample_next_token(
            logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_tokens=idx,
        )

        # Append sampled token to sequence
        idx = torch.cat((idx, next_token), dim=1)

        # Stop early if batch size is 1 and EOS token was generated
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return idx


def generate_text(
    model: TinyLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.1,
    device: Optional[torch.device] = None,
) -> str:
    """
    High-level convenience function: takes a string prompt and returns generated string.
    """
    if device is None:
        device = next(model.parameters()).device

    prompt_tokens = tokenizer.encode(prompt)
    if len(prompt_tokens) == 0:
        prompt_tokens = [tokenizer.encode("<|endoftext|>")[0]]

    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    eos_id = tokenizer.encode("<|endoftext|>")[0]

    out_idx = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_id,
    )

    generated_ids = out_idx[0].tolist()
    return tokenizer.decode(generated_ids)
