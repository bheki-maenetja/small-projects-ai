import torch.optim as optim

def get_optimiser(opt_name, model, l_rate, **kwargs):
    if opt_name == "adam":
        return optim.AdamW(model.parameters(), lr=l_rate, **kwargs)
    elif opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=l_rate, **kwargs)
    elif opt_name == "rms":
        return optim.RMSprop(model.parameters(), lr=l_rate, **kwargs)
    elif opt_name == "lbfgs":
        return optim.LBFGS(model.parameters(), lr=l_rate, **kwargs)