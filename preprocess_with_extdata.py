# Python stdlib
from pathlib import Path
import random
import json
import os
# General DS
import numpy as np
import pandas as pd
import multiprocessing as mp
from transformers import AutoTokenizer

from src.utils import nice_pbar, make_folder
from src.data.preprocess import nb_to_df


MAX_N_CELLS = 126
MODEL_NAME = 'microsoft/deberta-v3-base'
TOKENIZER_PATH = 'microsoft/deberta-v3-base'
RAW_DIR: str = Path(os.environ['RAW_DIR'])
PROC_DIR: str = Path(os.environ['PROC_DIR'])
PCT_DATA: str = float(os.environ['PCT_DATA'])
#PCT_DATA = 0.01

# This block which I originally added as debug has saved me so many times... kep forgetting to source env
if not make_folder(PROC_DIR):
    print("""
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
df['is_code'] = (df['cell_type'] == 'code').astype(np.int8)
df['pos'] = df.groupby('id')['cell_id'].cumcount() + 1  # [1:MAX_N_CELLS]
# dummy start has 0 real_pos, code cells have pos/n_codes
# last code cells, rel_pos = 1
df['rel_pos'] = df['pos'] / df.groupby('id')['is_code'].transform('sum')
#df.loc[df['cell_type'] == 'mark', 'pos'] = 'null'
df.loc[df['cell_type'] == 'mark', 'rel_pos'] = 0

# Add cell type and cell index
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
df['source'] = (
    df['cell_type']
    + tokenizer.sep_token 
    + df['source']
)

df_orders = pd.read_csv(RAW_DIR / 'train_orders.csv')
df_orders['cell_order'] = df_orders['cell_order'].str.split()
df_orders = df_orders.explode('cell_order')
df_orders['rank'] = df_orders.groupby('id')['cell_order'].cumcount()
df_orders['pct_rank'] = (
    df_orders['rank'] / df_orders.groupby('id')['cell_order'].transform('count')
)
df_orders.rename(columns={'cell_order': 'cell_id'}, inplace=True)
df_merge = df.merge(df_orders, how='left', on=['id', 'cell_id'])

# Process external data
df_ext = pd.read_pickle(RAW_DIR / 'external_dataset_processed.pickle')
n_ids = df_ext.groupby('source')['id'].transform('nunique')
last_id = df_ext.groupby('source')['id'].transform('last')
df_ext['is_dup_md'] = df_ext['is_dup'] = (n_ids > 1) & (df_ext['id'] != last_id)
df_ext.loc[df_ext['cell_type'] == 'code', 'is_dup_md'] = np.nan
dup_pct = df_ext.groupby('id').agg({
    'is_dup': 'mean',
    'is_dup_md': 'mean'
})
non_dup_notebooks = dup_pct.loc[
    (dup_pct['is_dup'] < 0.5) &
    (dup_pct['is_dup_md'] < 0.5)
].index.tolist()


df_ext = df_ext.loc[df_ext['id'].isin(non_dup_notebooks)].copy()
df_ext['id'] = df_ext['id'].astype(str)
df_ext.loc[df_ext['cell_type'] == 'markdown', 'cell_type'] = 'mark'
df_ext = df_ext.sort_values(['id', 'cell_type', 'rank']).copy()
df_ext['cell_id'] = np.arange(1, df_ext.shape[0] + 1)
df_ext['cell_id'] = df_ext['cell_id'].astype(str)
df_ext['is_code'] = (df_ext['cell_type'] == 'code').astype(np.int8)
df_ext['pos'] = df_ext.groupby('id')['cell_id'].cumcount() + 1  # [1:MAX_N_CELLS]
df_ext['rel_pos'] = df_ext['pos'] / df_ext.groupby('id')['is_code'].transform('sum')
#df_ext.loc[df_ext['cell_type'] == 'mark', 'pos'] = 'null'
df_ext.loc[df_ext['cell_type'] == 'mark', 'rel_pos'] = 0

df_ext['source'] = (
    df_ext['cell_type']
    + tokenizer.sep_token 
    + df_ext['source']
)
df_ext = df_ext[['cell_id', 'cell_type', 'source', 'id', 'is_code', 'pos', 'rel_pos', 'rank', 'pct_rank']].copy()

print('Kaggle data shape, external data shape:', df_merge.shape, df_ext.shape)
df_merge = pd.concat([df_ext, df_merge])
print('Final data shape:', df_merge.shape)

df_merge[[
    'id', 
    'cell_id', 
    'cell_type', 
    'is_code',
    'pos', 
    'rel_pos',
    'source', 
    'rank', 
    'pct_rank', 
]].set_index(['id', 'cell_type']).to_pickle(PROC_DIR / 'cells.pkl')

df_merge['n_codes'] = (df_merge['cell_type'] == 'code').astype(np.int8)
df_merge['n_mds'] = (df_merge['cell_type'] == 'mark').astype(np.int8)

df_nb = df_merge.groupby('id', as_index=False).agg({
    'cell_id': 'count',
    'n_codes': 'sum',
    'n_mds': 'sum'
}).rename(columns={'cell_id': 'n_cells'})
df_nb['md_pct'] = df_nb['n_mds'] / df_nb['n_cells']
#df_ancestors = pd.read_csv(RAW_DIR / 'train_ancestors.csv', index_col='id')
#df_nb['ancestor_id'] = df_nb['id'].map(df_ancestors['ancestor_id'])

# A dict for all notebook metadata
nb_meta = df_nb.set_index('id').to_dict(orient='index')
for d in nb_meta.values():
    d['n_codes'] = int(d['n_codes'])
    d['n_mds'] = int(d['n_mds'])
json.dump(
    nb_meta, open(PROC_DIR / "nb_meta.json","wt")
)


val_set = pd.read_pickle(RAW_DIR / "val_id_1000.pkl")
train_df = df_nb.loc[~df_nb.id.isin(val_set), 'id'].reset_index(drop=True)

train_df = train_df[
    train_df.map(lambda x: nb_meta[x]['n_cells']) <= MAX_N_CELLS
].reset_index(drop=True)

train_df.to_pickle(PROC_DIR / 'train_id.pkl')
val_set.to_pickle(PROC_DIR / 'val_id.pkl')
