import torch

def build_optimiser(model, config):
    if config.training.optimizer.name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.training.optimizer.lr, weight_decay=config.training.optimizer.weight_decay)

    elif config.training.optimizer.name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.training.optimizer.lr, weight_decay=config.training.optimizer.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer.name}")