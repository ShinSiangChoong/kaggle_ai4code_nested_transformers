import json
import pandas as pd
from torch.utils.data import DataLoader

from src.data.datasets.baseline_data import MarkdownDataset


def get_dl(is_train, args) -> DataLoader:
    """Get train data loader
    """
    df_ids = (
        pd.read_pickle(args.train_id_path) 
        if is_train else pd.read_pickle(args.val_id_path)
    )
    df_codes = pd.read_pickle(args.codes_path)
    df_mds = pd.read_pickle(args.mds_path)
    df_mds = df_mds.loc[df_mds['id'].isin(df_ids)].reset_index(drop=True)
    nb_meta = json.load(open(args.nb_meta_path, "rt"))

    ds = MarkdownDataset(
        df_ids=df_ids,
        df_codes=df_codes,
        df_mds=df_mds,
        nb_meta=nb_meta,
        model_name_or_path=args.model_name_or_path,
        max_n_codes=args.max_n_codes,
        max_md_len=args.max_md_len,
        max_len=args.max_len
    )
    data_loader = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=is_train, 
        num_workers=args.n_workers,
        pin_memory=False, 
        drop_last=is_train
    )
    return data_loader
