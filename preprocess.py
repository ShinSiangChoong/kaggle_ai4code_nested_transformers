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


RAW_DIR: str = Path(os.environ['RAW_DIR'])
PROC_DIR: str = Path(os.environ['PROC_DIR'])
PCT_DATA: str = float(os.environ['PCT_DATA'])
# PCT_DATA = 0.0001

MODEL_NAME = 'microsoft/codebert-base'

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
df['code_idx'] = (df['cell_type'] == 'code').astype(np.int8)
df['code_idx'] = df.groupby('id')['code_idx'].cumsum().astype(str)

# Add a special number token for code cell to shows its rank within code cells
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
df.loc[df['cell_type'] == 'code', 'source'] = (
    df.loc[df['cell_type'] == 'code', 'code_idx'] 
    + tokenizer.sep_token + df.loc[df['cell_type'] == 'code', 'source']
)

df_orders = pd.read_csv(RAW_DIR / 'train_orders.csv')
df_orders['cell_order'] = df_orders['cell_order'].str.split()
df_orders = df_orders.explode('cell_order')
df_orders['rank'] = df_orders.groupby('id')['cell_order'].cumcount()
df_orders['rank_pct'] = (
    df_orders['rank'] / df_orders.groupby('id')['cell_id'].transform('count')
)
df_orders.rename(columns={'cell_order': 'cell_id'}, inplace=True)

df_merge = df.merge(df_orders, how='left', on=['id', 'cell_id'])
df_merge.loc[
    df_merge['cell_type'] == 'markdown', 
    ['source', 'id', 'rank', 'rank_pct']
].reset_index(drop=True).to_pickle(PROC_DIR / 'mds.pickle')
df_merge.loc[
    df_merge['cell_type'] == 'code', 
    ['source', 'id', 'code_idx', 'rank', 'rank_pct']
].set_index('id').to_pickle(PROC_DIR / 'codes.pickle')

df_merge['n_codes'] = (df_merge['cell_type'] == 'code').astype(np.int8)
df_merge['n_mds'] = (df_merge['cell_type'] == 'markdown').astype(np.int8)
df_nb = df_merge.groupby('id', as_index=False).agg({
    'cell_id': 'count',
    'n_codes': 'sum',
    'n_mds': 'sum'
}).rename(columns={'cell_id': 'n_cells'})
df_nb['md_pct'] = df_nb['n_mds'] / df_nb['n_cells']
df_ancestors = pd.read_csv(RAW_DIR / 'train_ancestors.csv', index_col='id')
df_nb['ancestor_id'] = df_nb['id'].map(df_ancestors['ancestor_id'])

# A dict for all notebook metadata
nb_meta = df_nb.drop('ancestor_id', axis=1).set_index('id').to_dict(orient='index')
for d in nb_meta.values():
    d['n_codes'] = int(d['n_codes'])
    d['n_mds'] = int(d['n_mds'])
json.dump(
    nb_meta, open(PROC_DIR / "nb_meta.json","wt")
)

NVALID = 0.1  # size of validation set
splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)
train_ind, val_ind = next(splitter.split(df_nb, groups=df_nb["ancestor_id"]))
train_df = df_nb.loc[train_ind, 'id'].reset_index(drop=True)
val_df = df_nb.loc[val_ind, 'id'].reset_index(drop=True)

train_df.to_pickle(PROC_DIR / 'train_id.pickle')
val_df.to_pickle(PROC_DIR / 'val_id.pickle')

df_merge.loc[
    df_merge['id'].isin(val_df),
    ['id', 'cell_type', 'cell_id', 'source', 'code_idx', 'rank', 'rank_pct']
].to_csv(PROC_DIR / 'val.csv')
print(df_merge.loc[
    df_merge['id'].isin(val_df),
    ['id', 'cell_type', 'cell_id', 'source', 'code_idx', 'rank', 'rank_pct']
].shape)
print(df_merge.loc[
    df_merge['id'].isin(val_df) & 
    (df_merge['cell_type'] == 'markdown'),
    ['id', 'cell_type', 'cell_id', 'source', 'code_idx', 'rank', 'rank_pct']
].shape)