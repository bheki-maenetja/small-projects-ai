import tiktoken

# Data loading
def get_text(name: str) -> str:
    """
    Read a text file and return its content as a string.
    """
    f_paths = {
        "shakespeare": "./data/shakespeare.txt",
        "jekyll": "./data/jekyll.txt",
        "gatsby": "./data/gatsby.txt",
        "dorian": "./data/dorian-gray.txt",
        "earth": "./data/theory-of-earth.txt",
    }
    f_path = f_paths.get(name, None)

    # Check if the file exists
    if f_path is None:
        raise ValueError(f"Unknown dataset name: {name}. Available options are: {list(f_paths.keys())}")
    
    # Read the file
    with open(f_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

# Vocabulary encoding and decoding
def get_vocab(text):
    """
    Get the vocabulary of a text as a sorted list of unique characters.
    """
    return sorted(list(set(text)))

def char_to_idx(chars=None, text=None):
    """
    Create a mapping from characters to indices.
    If chars is not provided, it will be generated from the text.
    """
    if chars is None:
        chars = get_vocab(text)
    return {ch:i for i, ch in enumerate(chars)}

def idx_to_char(chars=None, text=None):
    """
    Create a mapping from indices to characters.
    If chars is not provided, it will be generated from the text.
    """
    if chars is None:
        chars = get_vocab(text)
    return {i:ch for i, ch in enumerate(chars)}

# Text manager
class TextManager:
    """A class to manage text encoding and decoding."""
    
    def __init__(self, text, enc_method="simple"):
        self.vocab = get_vocab(text)
        self.vocab_size = len(self.vocab)
        self.c_to_i = char_to_idx(self.vocab)
        self.i_to_c = idx_to_char(self.vocab)
        self.enc_method = enc_method
        
        if enc_method == "tiktoken":
            self.enc = tiktoken.get_encoding("gpt2")
        else:
            self.enc = None
        
    def __str__(self):
        return f"""Vocabulary (size = {self.vocab_size}):
        {"".join(self.vocab)}
        """

    def get_vocab(self, as_str=False):
        if as_str:
            return "".join(self.vocab)
        return self.vocab
    
    def encode(self, text):
        """
        Take a string, output a list of integers.
        """
        if self.enc_method == "tiktoken":
            return self.enc.encode(text)
        return [self.c_to_i[c] for c in text]
    
    def decode(self, indices):
        """
        Take a list of integers, output a string.
        """
        if self.enc_method == "tiktoken":
            return self.enc.decode(indices)
        return "".join([self.i_to_c[idx] for idx in indices])