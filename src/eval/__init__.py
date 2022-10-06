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
    with torch.inference_mode():
        for d in pbar:
            nb_ids.extend(d['nb_ids'])
            for k in d:
                if k != 'nb_ids':
                    d[k] = d[k].cuda()
            with torch.cuda.amp.autocast(False):
                point_pred = model(
                    d['tokens'], 
                    d['cell_masks'], 
                    d['cell_fea'],
                    d['nb_atn_masks'], 
                    d['md_pct'],
                    d['next_masks'],
                    is_train=False 
                )
            indices = torch.where(d['nb_reg_masks'] == 1)
            point_preds.extend(point_pred[indices].cpu().numpy().tolist())
    return nb_ids, point_preds


def get_point_preds(point_preds: np.array, df: pd.DataFrame):
    df = df.reset_index()
    df['preds'] = point_preds
    df['pred_rank'] = df.groupby('id')['preds'].rank()
    code_rank_correction(df)
    return df.sort_values('pp_rank').groupby('id')['cell_id'].apply(list)


def get_pair_kernel_preds(kernel_pairs: np.array, df: pd.DataFrame):
    """
    Make use of pairhead predictions with the logic in the public notebook:
    https://www.kaggle.com/code/yuanzhezhou/ai4code-pairwise-bertsmall-inference
    """
    df = df.reset_index()
    df["pred_pair_kernel"] = kernel_pairs
    df.loc[df["cell_type"] == "code", "pred_point_kernel"] = df.loc[
        df["cell_type"] == "code", "pos"
    ]
    pred_pair_kernel = df.sort_values('pred_pair_kernel').groupby('id')['cell_id'].apply(list)
    return pred_pair_kernel


def code_rank_correction(df):
    """Swap the code cells based on the given order
    """
    df['pp_rank'] = df['pred_rank'].copy()
    df.loc[df['cell_type'] == 'code', 'pp_rank'] = df.loc[
        df['cell_type'] == 'code'
    ].sort_values(['id', 'preds'])['pred_rank'].values
    print('non-corrected %:', (df['pp_rank'] == df['pred_rank']).mean())
