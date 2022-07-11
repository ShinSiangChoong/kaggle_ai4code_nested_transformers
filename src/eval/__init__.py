from collections import defaultdict

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

from src.utils import nice_pbar
from src.data import read_data


def get_preds(model: nn.Module, loader: DataLoader, df: pd.DataFrame):
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
    nb_ids = []
    preds = []

    with torch.inference_mode():
        for idx, d in enumerate(pbar):
            nb_ids.extend(d['nb_id'])
            for k in d:
                if k != 'nb_id':
                    d[k] = d[k].cuda()
            with torch.cuda.amp.autocast():
                pred = model(
                    d['tokens'], 
                    d['cell_masks'], 
                    d['nb_masks'], 
                    d['label_masks'],
                    d['pos'],
                    d['start'], 
                    d['md_pct']
                ).masked_fill(d['label_masks'], -6.5e4)
            preds.append(F.softmax(pred, dim=-1).detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        res = defaultdict(list)
        for nb_id, pred in zip(nb_ids, preds):
            cell_ids = df.loc[nb_id, 'cell_id'].values
            r, c = linear_sum_assignment(1 - pred)
            curr2next = dict(zip(r, c))
            curr = 0
            while curr <= len(cell_ids):
                curr = curr2next[curr]
                res[nb_id].append(cell_ids[curr-1])
    return res
