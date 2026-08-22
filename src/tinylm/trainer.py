import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tinylm.config import TinyLMConfig, get_device
from tinylm.model import TinyLM


@dataclass
class TrainerConfig:
    """
    Hyperparameters for the training loop.
    """
    learning_rate: float = 5e-4       # Peak learning rate
    min_lr: float = 5e-5              # Minimum learning rate after cosine decay (10% of peak)
    weight_decay: float = 0.1         # Decoupled weight decay for 2D parameters
    beta1: float = 0.9                # AdamW beta1
    beta2: float = 0.95               # AdamW beta2 (0.95 is standard for LLMs, vs 0.999 in vision)
    grad_clip: float = 1.0            # Max gradient norm for clipping
    
    warmup_steps: int = 100           # Steps of linear warmup
    max_steps: int = 1000             # Total training steps
    
    eval_interval: int = 100          # Evaluate validation loss every N steps
    eval_iters: int = 20              # How many batches to average for validation loss
    checkpoint_dir: str = "checkpoints"  # Directory to save model checkpoints


def configure_optimizers(
    model: TinyLM,
    weight_decay: float = 0.1,
    learning_rate: float = 5e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    device_type: str = "mps",
) -> torch.optim.AdamW:
    """
    Separate model parameters into two groups:
    1. Decay: 2D weights (Linear layers, embeddings) -> apply weight decay
    2. No-Decay: 1D parameters (LayerNorm scales, biases) -> 0.0 weight decay
    """
    decay_params = []
    nodecay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Any parameter with >= 2 dimensions gets weight decay (matrices)
        # 1D parameters (biases, layernorm weights) do NOT get weight decay
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        fused=(device_type == "cuda"),  # Fused AdamW if on CUDA
    )
    return optimizer


def get_lr(step: int, config: TrainerConfig) -> float:
    """
    Compute learning rate at step `step` using Linear Warmup + Cosine Decay.
    """
    # 1) Linear warmup phase
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / (config.warmup_steps + 1)

    # 2) If step exceeds max_steps, return min_lr
    if step > config.max_steps:
        return config.min_lr

    # 3) Cosine decay phase down to min_lr
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


class Trainer:
    """
    Trainer orchestrates training steps, learning rate scheduling,
    gradient clipping, validation, and checkpointing.
    """

    def __init__(
        self,
        model: TinyLM,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainerConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config if config is not None else TrainerConfig()
        self.device = device if device is not None else get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Infinite train loader iterator
        self._train_iter = iter(self.train_loader)

        # Configure optimizer with decoupled parameter groups
        self.optimizer = configure_optimizers(
            self.model,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            device_type=self.device.type,
        )

        self.step = 0
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_next_train_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch the next batch from the training loader, looping indefinitely.
        """
        try:
            x, y = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            x, y = next(self._train_iter)
        return x.to(self.device), y.to(self.device)

    def train_step(self) -> float:
        """
        Execute a single forward-backward-optimizer step.
        """
        self.model.train()
        
        # 1. Update learning rate according to schedule
        lr = get_lr(self.step, self.config)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # 2. Get batch
        x, y = self._get_next_train_batch()

        # 3. Forward pass & loss
        self.optimizer.zero_grad(set_to_none=True)
        _, loss = self.model(x, targets=y)

        # 4. Backward pass
        loss.backward()

        # 5. Gradient clipping
        if self.config.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        # 6. Optimizer step
        self.optimizer.step()
        self.step += 1

        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate average loss over validation batches.
        """
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        val_iter = iter(self.val_loader)
        iters = min(self.config.eval_iters, len(self.val_loader))

        for _ in range(iters):
            try:
                x, y = next(val_iter)
            except StopIteration:
                break
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, targets=y)
            total_loss += loss.item()

        return total_loss / max(iters, 1)

    def save_checkpoint(self, filename: str = "tinylm_checkpoint.pt") -> Path:
        """
        Save model weights, optimizer state, config, and training progress.
        """
        filepath = self.checkpoint_dir / filename
        checkpoint = {
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.config.__dict__,
            "trainer_config": self.config.__dict__,
        }
        torch.save(checkpoint, filepath)
        return filepath

    def load_checkpoint(self, filepath: Path) -> None:
        """
        Load weights, optimizer state, and step from a checkpoint.
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]

    def fit(self) -> Dict[str, Any]:
        """
        Run the complete training loop for max_steps.
        """
        history = {"step": [], "train_loss": [], "val_loss": [], "lr": []}

        for step in range(self.step, self.config.max_steps):
            train_loss = self.train_step()
            current_lr = get_lr(step, self.config)

            if step % self.config.eval_interval == 0 or step == self.config.max_steps - 1:
                val_loss = self.evaluate()
                history["step"].append(step)
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["lr"].append(current_lr)

                print(
                    f"Step {step:5d}/{self.config.max_steps:5d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"LR: {current_lr:.6f}"
                )

        # Save final checkpoint
        self.save_checkpoint("tinylm_final.pt")
        return history
