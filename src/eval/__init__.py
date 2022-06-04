import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils import nice_pbar
from src.data import read_data


def get_preds(model: nn.Module, loader: DataLoader):
    """Get labels and predictions

    Args:
        model (nn.Module)
        loader (DataLoader)
    Returns:
        labels (np.array)
        preds (np.array)
    """
    model.eval()

    pbar = nice_pbar(loader, len(loader), "Validation")

    labels = []
    preds = []

    with torch.inference_mode():
        for idx, data in enumerate(pbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)
