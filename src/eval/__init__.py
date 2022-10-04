import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import nice_pbar


def get_raw_preds(model: nn.Module, loader: DataLoader):
    model.eval()
    pbar = nice_pbar(loader, len(loader), "Validation")    
    nb_ids = []
    point_preds = []
    pair_preds = []
    pair_preds_kernel = []
    with torch.inference_mode():
        for d in pbar:
            nb_ids.extend(d['nb_ids'])
            for k in d:
                if k != 'nb_ids':
                    d[k] = d[k].cuda()
            with torch.cuda.amp.autocast(False):
                point_pred, pair_pred = model(
                    d['tokens'], 
                    d['cell_masks'], 
                    d['cell_fea'],
                    d['nb_atn_masks'], 
                    d['md_pct'],
                    d['next_masks'], 
                )
            indices = torch.where(d['nb_reg_masks'] == 1)
            point_preds.extend(point_pred[indices].cpu().numpy().tolist())
            pair_preds.append(pair_pred.cpu().numpy())
            pair_pred_kernel = pair_pred.masked_fill(d['md2code_masks'], -6.5e4)
            pair_pred_kernel = F.softmax(pair_pred_kernel, dim=-1)
            pair_pred_kernel *= (torch.arange(point_pred.shape[1]+1).cuda()+1)
            pair_pred_kernel = pair_pred_kernel.sum(dim=-1)
            pair_preds_kernel.append(pair_pred_kernel[indices[0], indices[1]+1].cpu())
        pair_preds_kernel = torch.cat(pair_preds_kernel).numpy()
    return nb_ids, point_preds, pair_preds, pair_preds_kernel


def get_point_preds(point_preds: np.array, df: pd.DataFrame):
    df = df.reset_index()
    df['pred_point_ss'] = point_preds
    df["pred_point_kernel"] = df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
    df.loc[df["cell_type"] == "mark", "pred_point_kernel"] = df.loc[
        df["cell_type"] == "mark", "pred_point_ss"
    ]
    pred_point_kernel = df.sort_values('pred_point_kernel').groupby('id')['cell_id'].apply(list)
    pred_point_ss = df.sort_values('pred_point_ss').groupby('id')['cell_id'].apply(list)
    return pred_point_kernel, pred_point_ss


def get_pair_kernel_preds(kernel_pairs: np.array, df: pd.DataFrame):
    df = df.reset_index()
    df["pred_pair_kernel"] = kernel_pairs
    df.loc[df["cell_type"] == "code", "pred_point_kernel"] = df.loc[
        df["cell_type"] == "code", "pos"
    ]
    pred_pair_kernel = df.sort_values('pred_pair_kernel').groupby('id')['cell_id'].apply(list)
    return pred_pair_kernel