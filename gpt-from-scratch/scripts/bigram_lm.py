# Third-Party Imports
import torch
import torch.nn as nn

# Model Class
class BigramLM(nn.Module):
    def __init__(self, vocab_size, n_embd=32, block_size=8, device="cpu"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
        self.device = device

    def forward(self, idx, targets=None):
        B, T = idx.shape # (B x T)
        tok_emb = self.embedding(idx) # (B x T x C)
        pos_emb = self.pos_embedding(torch.arange(T, device=self.device)) # (T x C)
        logits = self.lm_head(tok_emb + pos_emb) # (B x T x vocab_size)

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
            # Get predictions
            logits, _ = self(idx)
            logits = logits[:, -1, :] # (B x C)
            
            # Apply softmax to probabilities
            probs = self.softmax(logits) # (B x C)

            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B x 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B x T+1)
        return idx