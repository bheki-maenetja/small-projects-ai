# Third-Party Imports
import torch
import torch.nn as nn

# Local Imports
from .attention_lm import AttentionHead

# Model Classes
class TransformerFeedForward(nn.Module):
    def __init__(self, n_embd=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerMultiHeadAttention(nn.Module):
    def __init__(self, n_heads=1, head_size=32, n_embd=32, block_size=8, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(
                head_size=head_size, 
                n_embd=n_embd, 
                block_size=block_size,
                dropout=dropout,
            ) 
            for _ in range(n_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class TransformerBlock(nn.Module):
    def __init__(self, n_embd=32, n_heads=1, block_size=8, dropout=0.1):
        super().__init__()
        self.sa = TransformerMultiHeadAttention(
            n_heads=n_heads, 
            head_size=n_embd//n_heads, 
            n_embd=n_embd, 
            block_size=block_size,
            dropout=dropout,
        )
        self.ffwd = TransformerFeedForward(n_embd=n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = self.sa(self.ln1(x)) + x
        x = self.ffwd(self.ln2(x)) + x
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, n_blocks=3, n_heads=4, n_embd=32, block_size=8, dropout=0.1, device="cpu"):
        super().__init__()
        self.block_size = block_size
        self.device = device

        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                n_embd=n_embd, 
                n_heads=n_heads,  
                block_size=block_size,
                dropout=dropout,
            )
            for _ in range(n_blocks)
        ], nn.LayerNorm(n_embd))
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, idx, targets=None):
        B, T = idx.shape # (B x T)
        tok_emb = self.embedding(idx) # (B x T x C)
        pos_emb = self.pos_embedding(torch.arange(T, device=self.device)) # (T x C)
        x = tok_emb + pos_emb # (B x T x C)
        x = self.blocks(x) # (B x T x C)
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
