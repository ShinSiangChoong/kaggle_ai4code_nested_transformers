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

    # labels = []
    preds = []

    with torch.inference_mode():
        for idx, d in enumerate(pbar):
            for k in d:
                if k != 'ids':
                    d[k] = d[k].cuda()
            with torch.cuda.amp.autocast():
                pred = model(d['tokens'], d['masks'], d['md_pct'], d['label_masks'])
            indices = np.where(d['label_masks'].cpu().numpy() == 1)
            pred = pred.detach().cpu().numpy()
            preds.append(pred[indices])
    return np.concatenate(preds)
