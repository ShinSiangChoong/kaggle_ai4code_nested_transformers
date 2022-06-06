# Python stdlib
from pathlib import Path
import random
import json
import os
# General DS
import numpy as np
import pandas as pd

from src.utils import nice_pbar, make_folder
from src.data.preprocess import nb_to_df


RAW_DIR: str = Path(os.environ['RAW_DIR'])
PROC_DIR: str = Path(os.environ['PROC_DIR'])
PCT_DATA: str = float(os.environ['PCT_DATA'])

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

pbar = nice_pbar(paths_train, total=len(paths_train), desc='Train NBs')
notebooks_train = [nb_to_df(path) for path in pbar]
df = (
    pd.concat(notebooks_train)
        .set_index('id', append=True)
        .swaplevel()
        .sort_index(level='id', sort_remaining=False)
)

df_orders = pd.read_csv(
    RAW_DIR / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()  # Split the string representation of cell_ids into a list


def get_ranks(base, derived):
    return [base.index(d) for d in derived]


df_orders_ = df_orders.to_frame().join(
    df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
    how='right',
)

ranks = {}
for id_, cell_order, cell_id in df_orders_.itertuples():
    ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}
df_ranks = (
    pd.DataFrame
        .from_dict(ranks, orient='index')
        .rename_axis('id')
        .apply(pd.Series.explode)
        .set_index('cell_id', append=True)
)

df_ancestors = pd.read_csv(RAW_DIR / 'train_ancestors.csv', index_col='id')
df = df.reset_index().merge(df_ranks, on=["id", "cell_id"]).merge(df_ancestors, on=["id"])
df["pct_rank"] = df["rank"] / df.groupby("id")["cell_id"].transform("count")

from sklearn.model_selection import GroupShuffleSplit

NVALID = 0.1  # size of validation set
splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)
train_ind, val_ind = next(splitter.split(df, groups=df["ancestor_id"]))
train_df = df.loc[train_ind].reset_index(drop=True)
val_df = df.loc[val_ind].reset_index(drop=True)

# Base markdown dataframes
train_df_mark = train_df[train_df["cell_type"] == "markdown"].reset_index(drop=True)
val_df_mark = val_df[val_df["cell_type"] == "markdown"].reset_index(drop=True)
train_df_mark.to_csv(PROC_DIR / "train_mark.csv", index=False)
val_df_mark.to_csv(PROC_DIR / "val_mark.csv", index=False)
val_df.to_csv(PROC_DIR / "val.csv", index=False)
train_df.to_csv(PROC_DIR / "train.csv", index=False)


# Additional code cells
def clean_code(cell):
    return str(cell).replace("\\n", "\n")


def sample_cells(cells, n):
    cells = [clean_code(cell) for cell in cells]
    if n >= len(cells):
        return [cell[:200] for cell in cells]
    else:
        results = []
        step = len(cells) / n
        idx = 0
        while int(np.round(idx)) < len(cells):
            results.append(cells[int(np.round(idx))])
            idx += step
        assert cells[0] in results
        if cells[-1] not in results:
            results[-1] = cells[-1]
        return results


def get_features(df: pd.DataFrame):
    features = dict()
    df = df.sort_values("rank").reset_index(drop=True)
    for idx, sub_df in nice_pbar(df.groupby("id"), df['id'].nunique(), "Get Feat"):
        features[idx] = dict()
        total_md = sub_df[sub_df.cell_type == "markdown"].shape[0]
        code_sub_df = sub_df[sub_df.cell_type == "code"]
        total_code = code_sub_df.shape[0]
        codes = sample_cells(code_sub_df.source.values, 20)
        features[idx]["total_code"] = total_code
        features[idx]["total_md"] = total_md
        features[idx]["codes"] = codes
    return features

val_fts = get_features(val_df)
json.dump(val_fts, open(PROC_DIR / "val_fts.json","wt"))
train_fts = get_features(train_df)
json.dump(train_fts, open(PROC_DIR / "train_fts.json","wt"))
