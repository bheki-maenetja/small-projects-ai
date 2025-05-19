import tiktoken
import torch

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
        self.text = text
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

        First 1000 characters:
        {self.text[:500]}
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
    
    def get_text_tensor(self):
        """
        Convert the text to a tensor of indices.
        """
        indices = self.encode(self.text)
        return torch.tensor(indices, dtype=torch.long)
    
    def get_text_tensor_split(self, test_size=0.1, print_dims=False):
        """
        Split the text into training and test sets.
        """
        data = self.get_text_tensor()
        n = int(data.shape[0] * (1 - test_size))
        train_tensor = data[:n]
        test_tensor = data[n:]

        if print_dims:
            print(f"Training Data ({train_tensor.shape})")
            print(train_tensor[:100])
            print(f"\nValidation Data ({test_tensor.shape})")
            print(test_tensor[:100])
        return train_tensor, test_tensor

# Batching
def create_batch(data, block_size=8, batch_size=4, device="cpu"):
    """
    Create a batch of data for training.
    """
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def batch_sanity_check(xb, yb):
    batch_size, block_size = xb.shape
    print("Inputs:", xb.shape, xb, sep="\n")
    print("Targets:", yb.shape, yb, sep="\n")
    print("=" * (3*block_size + 20))

    for b in range(batch_size): # batch dimension
        for bl in range(block_size): # block (time) dimension
            context = xb[b, :bl+1]
            target = yb[b, bl]
            print(f"When input (context) is {context.tolist()} target = {target}.")