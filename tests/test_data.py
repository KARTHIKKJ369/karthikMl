import pytest
import torch
from tinylm.data import TextDataset, create_dataloader


def test_text_dataset_slices():
    # Sequence of 10 tokens: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    tokens = list(range(10))
    context_length = 4
    stride = 4

    dataset = TextDataset(tokens, context_length=context_length, stride=stride)
    
    # Total chunks: (10 - 4) // 4 = 1 chunk (indices 0..4)
    assert len(dataset) == 1
    
    x, y = dataset[0]
    # x should be [0, 1, 2, 3]
    # y should be [1, 2, 3, 4]
    assert torch.equal(x, torch.tensor([0, 1, 2, 3], dtype=torch.long))
    assert torch.equal(y, torch.tensor([1, 2, 3, 4], dtype=torch.long))


def test_text_dataset_sliding_stride():
    tokens = list(range(10))
    context_length = 3
    stride = 1  # 1-token sliding window

    dataset = TextDataset(tokens, context_length=context_length, stride=stride)
    # Total samples: (10 - 3) // 1 = 7 samples
    assert len(dataset) == 7

    x0, y0 = dataset[0]
    assert torch.equal(x0, torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(y0, torch.tensor([1, 2, 3], dtype=torch.long))

    x1, y1 = dataset[1]
    assert torch.equal(x1, torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.equal(y1, torch.tensor([2, 3, 4], dtype=torch.long))


def test_dataloader_batch_shapes():
    # 100 dummy tokens
    tokens = torch.arange(100, dtype=torch.long)
    batch_size = 4
    context_length = 8

    loader = create_dataloader(tokens, batch_size=batch_size, context_length=context_length)
    
    batch_x, batch_y = next(iter(loader))
    
    assert batch_x.shape == (batch_size, context_length)
    assert batch_y.shape == (batch_size, context_length)
    assert batch_x.dtype == torch.long
    assert batch_y.dtype == torch.long

    # Verify target is always shifted by 1 relative to input for every row
    for b in range(batch_size):
        assert torch.equal(batch_x[b, 1:], batch_y[b, :-1])
