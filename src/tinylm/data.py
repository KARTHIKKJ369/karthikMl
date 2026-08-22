from typing import List, Tuple, Union, Optional
import urllib.request
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """
    PyTorch Dataset that slices a 1D tensor of token IDs into (x, y) pairs
    for autoregressive language modeling.
    
    For a context_length T:
        x (input)  = tokens[i : i + T]
        y (target) = tokens[i + 1 : i + T + 1]
    """

    def __init__(
        self,
        tokens: Union[torch.Tensor, List[int]],
        context_length: int,
        stride: Optional[int] = None,
    ) -> None:
        """
        Args:
            tokens: 1D tensor or list of integer token IDs.
            context_length: Length of the input context window (T).
            stride: Step size between consecutive chunks. 
                    Defaults to context_length (non-overlapping chunks).
        """
        super().__init__()
        if isinstance(tokens, list):
            self.tokens = torch.tensor(tokens, dtype=torch.long)
        else:
            self.tokens = tokens.to(dtype=torch.long)

        self.context_length = context_length
        # By default, stride = context_length means non-overlapping consecutive chunks
        self.stride = stride if stride is not None else context_length

        if len(self.tokens) <= context_length:
            raise ValueError(
                f"Token count ({len(self.tokens)}) must be greater than context_length ({context_length})"
            )

    def __len__(self) -> int:
        """
        Total number of sample pairs (x, y) that can be extracted.
        We need (context_length + 1) tokens for each sample.
        """
        return (len(self.tokens) - self.context_length) // self.stride

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch the idx-th sample pair (x, y).
        
        Returns:
            x: Input token IDs of shape (context_length,)
            y: Target token IDs of shape (context_length,) shifted right by 1
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.context_length

        # Input: tokens from start_idx to end_idx - 1 (length = context_length)
        x = self.tokens[start_idx:end_idx]
        # Target: tokens from start_idx + 1 to end_idx (length = context_length)
        y = self.tokens[start_idx + 1 : end_idx + 1]

        return x, y


def create_dataloader(
    tokens: Union[torch.Tensor, List[int]],
    batch_size: int,
    context_length: int,
    shuffle: bool = True,
    stride: Optional[int] = None,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader yielding batches of shape:
        x: (batch_size, context_length)
        y: (batch_size, context_length)
    """
    dataset = TextDataset(tokens, context_length=context_length, stride=stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def load_tinyshakespeare(data_dir: str = "data") -> str:
    """
    Download and return the TinyShakespeare dataset (~1MB of text).
    Cached locally in `data_dir/tinyshakespeare.txt`.
    """
    path = Path(data_dir) / "tinyshakespeare.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading TinyShakespeare dataset to {path}...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")
    
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
