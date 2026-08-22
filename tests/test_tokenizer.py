from tinylm.tokenizer import Tokenizer


def test_tokenizer_roundtrip():
    tokenizer = Tokenizer()
    text = "Hello, world! TinyLM is learning to speak."
    tokens = tokenizer.encode(text)
    
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert len(tokens) > 0
    
    decoded = tokenizer.decode(tokens)
    assert decoded == text


def test_tokenizer_vocab_size():
    tokenizer = Tokenizer()
    assert tokenizer.vocab_size == 50257


def test_tokenizer_special_token():
    tokenizer = Tokenizer()
    text = "Hello<|endoftext|>World"
    tokens = tokenizer.encode(text)
    assert tokenizer.decode(tokens) == text
