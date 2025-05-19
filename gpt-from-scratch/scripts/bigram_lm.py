# Third-Party Imports
import torch
import torch.nn as nn

# Local Imports
from .model_helpers import get_optimiser
from .data_handling import create_batch

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

# Training Function
def train(
    model, 
    train_data, 
    val_data, 
    max_iters=10000,
    eval_interval=1000,
    eval_iters=100,
    block_size=8,
    batch_size=32, 
    l_rate=1e-03, 
    device="cpu"
):
    model.to(device)
    opt = get_optimiser("adam", model, l_rate)
    
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            model.eval()
            out = {}
            with torch.no_grad():
                for split in ["train", "val"]:
                    losses = torch.zeros(eval_iters)
                    for i in range(eval_iters):
                        xb, yb = create_batch(
                            train_data if split == "train" else val_data,
                            block_size=block_size, 
                            batch_size=batch_size, 
                            device=device
                        )
                        logits, loss = model(xb, yb)
                        losses[i] = loss.item()
                    out[split] = losses.mean().item()
                print(f"Step: {iter}, Train Loss: {out['train']:.4f}, Val Loss: {out['val']:.4f}")

        # Training
        model.train()
        xb, yb = create_batch(
            train_data,
            block_size=block_size, 
            batch_size=batch_size, 
            device=device
        )

        # Calculate loss and backpropagate
        logits, loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

# Text Generation
def generate_text(start_char, tm, model, max_new_tokens=100, device="cpu"):
    start_idx = tm.encode(start_char)
    start_idx = torch.tensor(start_idx, dtype=torch.long, device=device).view((1,1))
    gen_idx = model.generate(start_idx, max_new_tokens)[0].tolist()
    return tm.decode(gen_idx)