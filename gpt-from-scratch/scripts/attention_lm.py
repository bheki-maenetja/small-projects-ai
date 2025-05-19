# Third-Party Imports
import torch
import torch.nn as nn

# Local Imports
from .model_helpers import get_optimiser
from .data_handling import create_batch

# Model Class
class AttentionHead(nn.Module):
    def __init__(self, head_size=16, n_embd=32, block_size=8):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("tril", torch.tril(torch.ones(1, 1, block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape # C = head_size
        k = self.key(x) # (B x T x head_size)
        q = self.query(x) # (B x T x head_size)

        # Com[pute attention scores
        wei = q @ k.transpose(-2, -1) / (C**-0.5) # (B x T x C) @ (B x C x T) -> (B x T x T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B x T x T)
        wei = wei.softmax(dim=-1) # (B x T x T)

        # Peform the weighted aggregation
        v = self.value(x) # (B x T x C)

        out = wei @ v # (B x T x T) @ (B x T x C) -> (B x T x C)
        return out

class AttentionLM(nn.Module):
    def __init__(self, vocab_size, n_embd=32, block_size=8, head_size=32, device="cpu"):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.head_size = head_size
        self.device = device

        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.sa_head = AttentionHead(head_size=head_size, n_embd=n_embd, block_size=block_size)

        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, idx, targets=None):
        B, T = idx.shape # (B x T)
        tok_emb = self.embedding(idx) # (B x T x C)
        pos_emb = self.pos_embedding(torch.arange(T, device=self.device)) # (T x C)
        logits = self.sa_head(tok_emb + pos_emb) # (B x T x C)
        logits = self.lm_head(logits) # (B x T x vocab_size)

        if targets is not None:
            # Reshape logits and targets
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
    
            # Calculate loss
            loss = self.loss(logits, targets)
        else:
            loss = None
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop the context to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]

            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # (B x C)
            
            # Apply softmax to probabilities
            probs = self.softmax(logits) # (B x C)

            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B x 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B x T+1)
        return idx