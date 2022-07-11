# Python stdlib
from pathlib import Path
import random
import json
import os
# General DS
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import multiprocessing as mp
from transformers import AutoTokenizer

from src.utils import nice_pbar, make_folder
from src.data.preprocess import nb_to_df


MAX_N_CELLS = 126
MODEL_NAME = 'microsoft/codebert-base'
RAW_DIR: str = Path(os.environ['RAW_DIR'])
PROC_DIR: str = Path(os.environ['PROC_DIR'])
PCT_DATA: str = float(os.environ['PCT_DATA'])
# PCT_DATA = 0.0001

# This block which I originally added as debug has saved me so many times... kep forgetting to source env
if not make_folder(PROC_DIR):
    print("""\
    Useful commands

    source env.sh
    env | grep DIR
    """)
    raise Exception(f'PROC DIR: {PROC_DIR} already exists. Did you forget to source env?')


paths_train = list((RAW_DIR / 'train').glob('*.json'))
random.seed(str(PROC_DIR))
random.shuffle(paths_train)

N = int(PCT_DATA * len(paths_train))
paths_train = paths_train[:N]

# Use multiprocess to improve speed
with mp.Pool(mp.cpu_count()) as p:
    notebooks_train = list(
        p.map(
            nb_to_df, 
            nice_pbar(paths_train, total=len(paths_train), desc='Train NBs')
        )
    )

df = pd.concat(notebooks_train).reset_index()
df.loc[df['cell_type'] == 'markdown', 'cell_type'] = 'mark'
df['cell_idx'] = df.groupby('id')['cell_id'].cumcount() # [0:MAX_N_CELLS-1]
df['pos'] = df['cell_idx'] + 1 # [1:MAX_N_CELLS]
df.loc[df['cell_type'] == 'mark', 'pos'] = MAX_N_CELLS + 1

# Add cell type and cell index
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
df['source'] = (
    df['cell_type'] 
    + tokenizer.sep_token 
    + df['source']
)

df_orders = pd.read_csv(RAW_DIR / 'train_orders.csv')
df_orders['cell_order'] = df_orders['cell_order'].str.split()
df_orders = df_orders.explode('cell_order')
df_orders['rank'] = df_orders.groupby('id')['cell_order'].cumcount()
df_orders['rank_pct'] = (
    df_orders['rank'] / df_orders.groupby('id')['cell_order'].transform('count')
)
df_orders.rename(columns={'cell_order': 'cell_id'}, inplace=True)
df_merge = df.merge(df_orders, how='left', on=['id', 'cell_id'])

df_tmp = df_merge.sort_values(['id', 'rank']).reset_index(drop=True)
df_tmp['next_cell_idx'] = df_tmp.groupby('id')['cell_idx'].shift(-1)
df_tmp['next_cell_idx'].fillna(
    df_tmp.groupby('id')['cell_idx'].transform(max)+1, inplace=True
)
df_tmp['unique_cell_id'] = (
    df_tmp['id'].astype(str) + '_' + df_tmp['cell_idx'].astype(str)
)
df_merge['next_cell_idx'] = (
    df_merge['id'].astype(str) + '_' + df_merge['cell_idx'].astype(str)
).map(df_tmp.set_index(['unique_cell_id'])['next_cell_idx']).astype(int)

df_merge[[
    'id', 
    'cell_id', 
    'cell_idx', 
    'cell_type', 
    'pos', 
    'source', 
    'rank', 
    'rank_pct', 
    'next_cell_idx'
]].set_index(['id', 'cell_type']).to_pickle(DATA_DIR / 'cells.pkl')

df_merge['n_codes'] = (df_merge['cell_type'] == 'code').astype(np.int8)
df_merge['n_mds'] = (df_merge['cell_type'] == 'mark').astype(np.int8)

df_nb = df_merge.groupby('id', as_index=False).agg({
    'cell_id': 'count',
    'n_codes': 'sum',
    'n_mds': 'sum'
}).rename(columns={'cell_id': 'n_cells'})
df_nb['md_pct'] = df_nb['n_mds'] / df_nb['n_cells']
df_ancestors = pd.read_csv(RAW_DIR / 'train_ancestors.csv', index_col='id')
df_nb['ancestor_id'] = df_nb['id'].map(df_ancestors['ancestor_id'])
df_nb['1st_cell_idx'] = df_nb['id'].map(df_tmp.groupby('id')['cell_idx'].first())

# A dict for all notebook metadata
nb_meta = df_nb.drop('ancestor_id', axis=1).set_index('id').to_dict(orient='index')
for d in nb_meta.values():
    d['n_codes'] = int(d['n_codes'])
    d['n_mds'] = int(d['n_mds'])
    d['1st_cell_idx'] = int(d['1st_cell_idx'])
json.dump(
    nb_meta, open(PROC_DIR / "nb_meta.json","wt")
)

NVALID = 0.1  # size of validation set
splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)
train_ind, val_ind = next(splitter.split(df_nb, groups=df_nb["ancestor_id"]))
train_df = df_nb.loc[train_ind, 'id'].reset_index(drop=True)
val_df = df_nb.loc[val_ind, 'id'].reset_index(drop=True)

train_df = train_df[
    train_df.map(lambda x: nb_meta[x]['n_cells']) <= MAX_N_CELLS
].reset_index(drop=True)
val_df = val_df[
    val_df.map(lambda x: nb_meta[x]['n_cells']) <= MAX_N_CELLS
].reset_index(drop=True)

train_df.to_pickle(PROC_DIR / 'train_id.pkl')
val_df.to_pickle(PROC_DIR / 'val_id.pkl')
