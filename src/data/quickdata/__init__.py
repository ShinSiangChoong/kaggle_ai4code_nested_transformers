import json
import pandas as pd
from torch.utils.data import DataLoader

from src.data.datasets.baseline_data import NotebookDataset


def get_dl(is_train, args) -> DataLoader:
    """Get train data loader
    """
    df_ids = (
        pd.read_pickle(args.train_id_path) 
        if is_train else pd.read_pickle(args.val_id_path)
    )
    df_cells = pd.read_pickle(args.cells_path)
    nb_meta = json.load(open(args.nb_meta_path, "rt"))

    ds = NotebookDataset(
        df_ids=df_ids,
        df_cells=df_cells,
        nb_meta=nb_meta,
        model_name_or_path=args.model_name_or_path,
        max_n_cells=args.max_n_cells,
        max_len=args.max_len,
        is_train=False
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
