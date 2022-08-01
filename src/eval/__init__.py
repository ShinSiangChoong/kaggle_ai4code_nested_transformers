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
            with torch.cuda.amp.autocast():
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
            pair_preds.append(pair_pred.cpu())
            pair_pred_kernel = pair_pred.masked_fill(d['md2code_masks'], -6.5e4)
            pair_pred_kernel = F.softmax(pair_pred_kernel, dim=-1)
            pair_pred_kernel *= (torch.arange(point_pred.shape[1]+1).cuda()+1)
            pair_pred_kernel = pair_pred_kernel.sum(dim=-1)
            pair_preds_kernel.append(pair_pred_kernel[indices[0], indices[1]+1].cpu())
        pair_preds_kernel = torch.cat(pair_preds_kernel).numpy()
    # return nb_ids, point_preds, pair_preds, pair_preds_kernel
    return nb_ids, point_preds, pair_preds_kernel


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


# def get_reg_preds(point_preds: np.array, df: pd.DataFrame, nb_meta: dict):
#     res = dict()
#     for nb_id, pred in nice_pbar(zip(nb_ids, preds.copy()), total=len(nb_ids), desc='Inference'):
#         cell_ids = df.loc[nb_id]['cell_id'].values
#         curr_code_idx = 0
#         curr_idx = 0
#         count = 0
#         tmp = []
#         while count < nb_meta[nb_id]['n_cells']:
#             mds = pred[curr_idx, nb_meta[nb_id]['n_codes']: nb_meta[nb_id]['n_cells']]
#             best_md = mds.argmax() + nb_meta[nb_id]['n_codes']
#             is_code = (
#                 pred[curr_idx][curr_code_idx] >= pred[curr_idx][best_md]
#                 and curr_code_idx < nb_meta[nb_id]['n_codes']
#             )
#             next_idx = int(curr_code_idx * is_code + best_md * (1 - is_code))
#             curr_code_idx += is_code
#             tmp.append(next_idx)
#             pred[:, next_idx] = 0
#             curr_idx = next_idx+1
#             count += 1
#         res[nb_id] = cell_ids[np.array(tmp)]

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
    # return pd.Series(res)
