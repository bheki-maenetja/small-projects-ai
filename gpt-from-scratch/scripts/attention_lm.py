# Third-Party Imports
import torch
import torch.nn as nn

# Model Classes
class FeedForward(nn.Module):
    def __init__(self, n_embd=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)

class AttentionHead(nn.Module):
    def __init__(self, head_size=16, n_embd=32, block_size=8, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape # C = head_size
        k = self.key(x) # (B x T x head_size)
        q = self.query(x) # (B x T x head_size)

        # Com[pute attention scores
        wei = q @ k.transpose(-2, -1) / (C**-0.5) # (B x T x C) @ (B x C x T) -> (B x T x T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B x T x T)
        wei = wei.softmax(dim=-1) # (B x T x T)
        wei = self.dropout(wei)
        # Peform the weighted aggregation
        v = self.value(x) # (B x T x C)

        out = wei @ v # (B x T x T) @ (B x T x C) -> (B x T x C)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=1, head_size=32, n_embd=32, block_size=8):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(head_size=head_size, n_embd=n_embd, block_size=block_size) 
            for _ in range(n_heads)
        ])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

class AttentionLM(nn.Module):
    def __init__(self, vocab_size, n_heads=1, head_size=32, n_embd=32, block_size=8, device="cpu"):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.head_size = head_size
        self.device = device

        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)

        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        if n_heads == 1:
            self.sa_heads = AttentionHead(head_size, n_embd, block_size)
        else:
            self.sa_heads = MultiHeadAttention(n_heads, n_embd//n_heads, n_embd, block_size)

        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, idx, targets=None):
        B, T = idx.shape # (B x T)
        tok_emb = self.embedding(idx) # (B x T x C)
        pos_emb = self.pos_embedding(torch.arange(T, device=self.device)) # (T x C)
        x = tok_emb + pos_emb # (B x T x C)
        x = self.sa_heads(x) # (B x T x C)
        x = self.ffwd(x) # (B x T x C)
        logits = self.lm_head(x) # (B x T x vocab_size)

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