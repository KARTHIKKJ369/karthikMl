import pytest
import torch
from tinylm.config import TinyLMConfig, get_device


def test_default_config():
    cfg = TinyLMConfig()
    assert cfg.vocab_size == 50257
    assert cfg.context_length == 256
    assert cfg.d_model == 256
    assert cfg.n_layers == 6
    assert cfg.n_heads == 8
    assert cfg.d_ff == 4 * 256  # 1024
    assert cfg.head_dim == 256 // 8  # 32


def test_invalid_heads():
    # d_model=100 is not divisible by n_heads=8, should raise ValueError
    with pytest.raises(ValueError, match="must be divisible"):
        TinyLMConfig(d_model=100, n_heads=8)


def test_presets():
    toy = TinyLMConfig.toy()
    assert toy.d_model == 64
    assert toy.head_dim == 32
    assert toy.n_layers == 2

    s = TinyLMConfig.tiny_s()
    assert s.d_model == 256
    assert s.n_layers == 6

    m = TinyLMConfig.tiny_m()
    assert m.d_model == 512
    assert m.n_layers == 8
    assert m.context_length == 512


def test_get_device():
    dev = get_device()
    assert isinstance(dev, torch.device)
    # On Apple Silicon M4, should detect 'mps'
    if torch.backends.mps.is_available():
        assert dev.type == "mps"
