# Third-Party Imports
import torch
import torch.optim as optim

# Local Imports
from .data_handling import create_batch

def get_optimiser(opt_name, model, l_rate, **kwargs):
    if opt_name == "adam":
        return optim.AdamW(model.parameters(), lr=l_rate, **kwargs)
    elif opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=l_rate, **kwargs)
    elif opt_name == "rms":
        return optim.RMSprop(model.parameters(), lr=l_rate, **kwargs)
    elif opt_name == "lbfgs":
        return optim.LBFGS(model.parameters(), lr=l_rate, **kwargs)
    
# Training Function
def train(
    model, 
    train_data, 
    val_data, 
    max_iters=10000,
    eval_interval=500,
    eval_iters=200,
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