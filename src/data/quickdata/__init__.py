import json
import pandas as pd
from torch.utils.data import DataLoader

from src.data.datasets.baseline_data import MarkdownDataset


def get_train_dl(args) -> DataLoader:
    """Get train data loader
    """

    train_df_mark = pd.read_csv(args.train_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
    train_fts = json.load(open(args.train_features_path))

    train_ds = MarkdownDataset(
        train_df_mark,
        model_name_or_path=args.model_name_or_path,
        md_max_len=args.md_max_len,
        code_max_len=args.code_max_len,
        total_max_len=args.total_max_len,
        fts=train_fts
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
        pin_memory=False, drop_last=True)

    return train_loader


def get_val_dl(args) -> DataLoader:
    """Get validation data loader
    """
    val_df_mark = pd.read_csv(args.val_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
    val_fts = json.load(open(args.val_features_path))

    val_ds = MarkdownDataset(
        val_df_mark,
        model_name_or_path=args.model_name_or_path,
        md_max_len=args.md_max_len,
        code_max_len=args.code_max_len,
        total_max_len=args.total_max_len,
        fts=val_fts
        )

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
        pin_memory=False, drop_last=False)

    return val_loader
