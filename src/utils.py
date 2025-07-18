import torch
import random
import numpy as np


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits, labels):
    # logits: [N, C]; labels: [N]
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct, labels.numel()


def gpu_mem_mb(device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        return 0.0
    return torch.cuda.memory_allocated(device) / (1024 ** 2)