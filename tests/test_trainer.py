import tempfile
from pathlib import Path
import pytest
import torch
from tinylm.config import TinyLMConfig, get_device
from tinylm.model import TinyLM
from tinylm.data import create_dataloader
from tinylm.trainer import Trainer, TrainerConfig, configure_optimizers, get_lr


def test_optimizer_parameter_groups():
    config = TinyLMConfig.toy()
    model = TinyLM(config)
    optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=1e-3)

    assert len(optimizer.param_groups) == 2
    decay_group = optimizer.param_groups[0]
    nodecay_group = optimizer.param_groups[1]

    assert decay_group["weight_decay"] == 0.1
    assert nodecay_group["weight_decay"] == 0.0

    # Ensure all decay params are 2D or higher (matrices)
    for p in decay_group["params"]:
        assert p.dim() >= 2
    # Ensure all nodecay params are 1D (biases, layernorms)
    for p in nodecay_group["params"]:
        assert p.dim() < 2


def test_lr_schedule():
    cfg = TrainerConfig(
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_steps=10,
        max_steps=100,
    )

    # At step 0, lr is near 0
    lr_0 = get_lr(0, cfg)
    assert lr_0 < 1e-3

    # At warmup_steps, lr is peak
    lr_warmup = get_lr(cfg.warmup_steps, cfg)
    assert abs(lr_warmup - 1e-3) < 1e-6

    # At halfway, lr is between peak and min
    lr_mid = get_lr(55, cfg)
    assert 1e-4 < lr_mid < 1e-3

    # At or past max_steps, lr is min_lr
    lr_end = get_lr(100, cfg)
    assert abs(lr_end - 1e-4) < 1e-6


def test_checkpoint_save_and_load():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = TinyLMConfig.toy()
        model = TinyLM(config)
        tokens = torch.randint(0, 100, (50,), dtype=torch.long)
        loader = create_dataloader(tokens, batch_size=2, context_length=8)

        trainer_cfg = TrainerConfig(max_steps=5, checkpoint_dir=tmp_dir)
        trainer = Trainer(model, train_loader=loader, config=trainer_cfg)

        # Train a step
        trainer.train_step()
        saved_path = trainer.save_checkpoint("test_ckpt.pt")
        assert saved_path.exists()

        # Load into a new model/trainer
        new_model = TinyLM(config)
        new_trainer = Trainer(new_model, train_loader=loader, config=trainer_cfg)
        new_trainer.load_checkpoint(saved_path)

        assert new_trainer.step == 1
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)


def test_overfit_single_batch():
    """
    CRITICAL TEST: Verify the model can completely overfit a tiny sequence.
    If backpropagation or attention is broken, loss will get stuck.
    If working, loss will drop rapidly to near zero (< 0.1).
    """
    device = get_device()
    config = TinyLMConfig.toy()
    config.dropout = 0.0  # Turn off dropout for overfitting test
    model = TinyLM(config).to(device)

    # A short repeated sequence: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] repeated
    tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10, dtype=torch.long)
    loader = create_dataloader(tokens, batch_size=2, context_length=8, shuffle=False)

    trainer_cfg = TrainerConfig(
        learning_rate=3e-3,
        min_lr=3e-3,
        weight_decay=0.0,
        warmup_steps=0,
        max_steps=120,
    )

    trainer = Trainer(model, train_loader=loader, config=trainer_cfg, device=device)

    initial_loss = trainer.train_step()
    for _ in range(119):
        final_loss = trainer.train_step()

    print(f"Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
    assert initial_loss > 4.0
    assert final_loss < 0.1, f"Model failed to overfit tiny batch! Final loss was {final_loss}"
