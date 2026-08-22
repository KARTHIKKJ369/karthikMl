from typing import List, Union
import tiktoken


class Tokenizer:
    """
    Tokenizer wrapper for TinyLM.
    Defaults to GPT-2 Byte-Pair Encoding (BPE) via tiktoken (vocab_size = 50257).
    """

    def __init__(self, encoding_name: str = "gpt2") -> None:
        """
        Initialize the tokenizer.
        'gpt2' uses byte-pair encoding with 50,257 vocabulary tokens.
        """
        self.encoding_name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)

    @property
    def vocab_size(self) -> int:
        """
        Return the total number of unique tokens in this tokenizer vocabulary.
        """
        return self._enc.n_vocab

    def encode(self, text: str) -> List[int]:
        """
        Convert a string into a list of integer token IDs.
        Example: "Hello, world!" -> [15496, 11, 995, 0]
        """
        # allowed_special={'<|endoftext|>'} allows encoding end-of-text special tokens
        return self._enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, token_ids: Union[List[int], List[List[int]]]) -> str:
        """
        Convert a list of integer token IDs back into a human-readable string.
        """
        # Handle single list of integers
        if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
            # Flatten if passed a 2D list (e.g. batch of size 1)
            token_ids = [item for sublist in token_ids for item in sublist]
        return self._enc.decode(token_ids)
