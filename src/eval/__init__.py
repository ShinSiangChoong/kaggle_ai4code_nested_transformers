import os

import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

from src.utils import nice_pbar
from src.data import read_data


def get_tsp_solution(M):
    # WRITE PROBLEM FILE
    f = open(f'group.par','w')
    f.write("PROBLEM_FILE = ../distances.sop\n")
    f.write("TOUR_FILE = ../output.txt\n")
    f.write(f"OPTIMUM = 0\n")
    f.write("MOVE_TYPE = 5\n")
    f.write("PATCHING_C = 3\n")
    f.write("PATCHING_A = 2\n")
    f.write("RUNS = 1\n")
    f.write("TIME_LIMIT = 120\n") #seconds
    f.close()

    # WRITE PARAMETER FILE
    f = open(f'distances.sop','w')
    f.write("NAME: distances\n")
    f.write("TYPE: SOP\n")
    f.write("COMMENT: SOP\n")
    f.write(f"DIMENSION: {len(M)}\n")
    f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
    f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
    f.write("EDGE_WEIGHT_SECTION\n")
    f.write(f"{len(M)}")
    for j in range(len(M)):
        #if j%25==0: print(j,', ',end='')
        for k in range(len(M)):
            f.write(f"{int(M[j,k])} ") 
        f.write("\n")
    f.close()
    
    # EXECUTE TSP SOLVER
    os.system("cd LKH-3.0.7; ./LKH ../group.par")# > /dev/null 2>&1")
    # !cd LKH-3.0.7; ./LKH ../group.par 
    
    # READ RESULTING ORDER
    with open('output.txt') as f:
        lines = f.readlines()
    for i,ln in enumerate(lines):
        if 'TOUR_SECTION' in ln: break
    perms = [int(x[:-1]) for x in lines[i+1:-2]]
    
    assert perms[0] == 1 and perms[-1] == len(M)
    os.system("rm group.par distances.sop output.txt")
    return np.array(perms[1:-1]) - 2


def get_preds(model: nn.Module, loader: DataLoader, df: pd.DataFrame, nb_meta: dict):
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
            nb_ids.extend(d['nb_ids'])
            for k in d:
                if k != 'nb_ids':
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

    res = dict()
    for nb_id, pred in nice_pbar(zip(nb_ids, preds.copy()), total=len(nb_ids), desc='Inference'):
        cell_ids = df.loc[nb_id]['cell_id'].values
        curr_code_idx = 0
        curr_idx = 0
        count = 0
        tmp = []
        while count < nb_meta[nb_id]['n_cells']:
            mds = pred[curr_idx, nb_meta[nb_id]['n_codes']: nb_meta[nb_id]['n_cells']]
            best_md = mds.argmax() + nb_meta[nb_id]['n_codes']
            is_code = (
                pred[curr_idx][curr_code_idx] >= pred[curr_idx][best_md]
                and curr_code_idx < nb_meta[nb_id]['n_codes']
            )
            next_idx = int(curr_code_idx * is_code + best_md * (1 - is_code))
            curr_code_idx += is_code
            tmp.append(next_idx)
            pred[:, next_idx] = 0
            curr_idx = next_idx+1
            count += 1
        res[nb_id] = cell_ids[np.array(tmp)]

    # res = dict()
    # for nb_id, pred in nice_pbar(zip(nb_ids, preds), total=len(nb_ids), desc='SOP'):
    #     cell_ids = df.loc[nb_id, 'cell_id'].values
    #     dm = 1 - pred[:len(cell_ids)+1, :len(cell_ids)+1]
    #     dm[dm == 1] = 6.5
    #     first_col = [6.5] * (len(cell_ids)+1)
    #     last_row = [6.5] * (len(cell_ids)+2)
    #     last_row[0] = 0
    #     dm = np.concatenate((np.expand_dims(first_col, axis=1), dm), axis=1)
    #     dm = np.concatenate((dm, np.expand_dims(last_row, axis=0)))
    #     dm *= 1e4
    #     dm = dm.astype(int)
    #     mask = np.ones(dm.shape)
    #     l = nb_meta[nb_id]['n_codes']+1
    #     mask[:l, :l] = np.triu(np.ones((l, l)))
    #     mask[:, 0] = 0    
    #     dm[np.where(mask == 0)] = -1
    #     dm[-1, 1:-1] = -1
    #     # dm[np.arange(len(dm)), np.arange(len(dm))] = 0
    #     perm = get_tsp_solution(dm)
    #     res[nb_id] = cell_ids[perm]
    #     break
    return pd.Series(res)
