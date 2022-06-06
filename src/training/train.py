# Python stdlib
from pathlib import Path
import argparse
import os
# General DS
import numpy as np
import pandas as pd
import wandb
# Torch
import torch
# Hugging Face
from transformers import AdamW, get_linear_schedule_with_warmup


from src.data.quickdata import get_train_dl, get_val_dl
from src.utils import nice_pbar, make_folder, lr_to_4sf
from src.eval.metrics import kendall_tau
from src.models.baseline import MarkdownModel
from src.data import read_data


def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base')
    # parser.add_argument('--train_mark_path', type=str, default='./data/train_mark.csv')
    # parser.add_argument('--train_features_path', type=str, default='./data/train_fts.json')
    # parser.add_argument('--val_mark_path', type=str, default='./data/val_mark.csv')
    # parser.add_argument('--val_features_path', type=str, default='./data/val_fts.json')
    # parser.add_argument('--val_path', type=str, default="./data/val.csv")

    parser.add_argument('--md_max_len', type=int, default=64)
    parser.add_argument('--total_max_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--n_workers', type=int, default=8)

    parser.add_argument('--wandb_mode', type=str, default="offline")
    parser.add_argument('--output_dir', type=str, default="./outputs")

    args = parser.parse_args()

    PROC_DIR = Path(os.environ['PROC_DIR'])
    args.train_mark_path = PROC_DIR / "train_mark.csv"
    args.train_features_path = PROC_DIR / "train_fts.json"
    args.val_mark_path = PROC_DIR / "val_mark.csv"
    args.val_features_path = PROC_DIR / "val_fts.json"
    args.val_path = PROC_DIR / "val.csv"
    return args


def train(model, train_loader, val_loader, epochs):
    np.random.seed(0)

    # TODO: Refactor to data
    DATA_DIR = Path(os.environ['RAW_DIR'])
    val_df = pd.read_csv(args.val_path)
    df_orders = pd.read_csv(
        DATA_DIR / 'train_orders.csv',
        index_col='id',
        squeeze=True,
    ).str.split()

    # Creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(args.epochs * len(train_loader) / args.accumulation_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    criterion = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_list = []
        # preds = []
        # labels = []

        tbar = nice_pbar(train_loader, len(train_loader), f"Train {epoch}")

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)
                loss = criterion(pred, target)
            scaler.scale(loss).backward()

            if idx % args.accumulation_steps == 0 or idx == len(tbar) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            loss_list.append(loss.item())
            # preds.append(pred.detach().numpy().ravel())
            # labels.append(target.detach().numpy().ravel())

            if idx % 20 == 0:
                metrics = {}
                metrics['loss'] = np.round(np.mean(loss_list[-20:]), 4)
                metrics['prev_lr'], metrics['next_lr'] = scheduler.get_last_lr()
                metrics['diff_lr'] = metrics['next_lr'] - metrics['prev_lr']
                wandb.log(metrics)

                tbar.set_postfix(loss=metrics['loss'], lr=lr_to_4sf(scheduler.get_last_lr()))

        torch.save(model.state_dict(), f"{args.output_dir}/model-{epoch}.bin")

        # TODO: Refactor to eval
        from src.eval import get_preds
        y_val, y_pred = get_preds(model, val_loader)
        val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
        val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred
        y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)

        print("Preds score", kendall_tau(df_orders.loc[y_dummy.index], y_dummy))
        print("Avg train loss", np.mean(loss_list))

    return model, y_pred


def main(args):
    make_folder(args.output_dir)

    train_loader = get_train_dl(args)
    val_loader = get_val_dl(args)

    model = MarkdownModel(args.model_name_or_path)
    model = model.cuda()
    wandb.watch(model, log_freq=10000, log_graph=True, log="all")
    model, y_pred = train(model, train_loader, val_loader, epochs=args.epochs)


if __name__ == '__main__':
    args = parse_args()
    wandb.init(
        project="AI4Code Dev",
        name="Baseline",
        mode=args.wandb_mode,
    )
    try:
        main(args)
    finally:
        wandb.finish()
