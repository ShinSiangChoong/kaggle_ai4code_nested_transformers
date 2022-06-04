# Python stdlib
from pathlib import Path
import argparse
# General DS
import numpy as np
import pandas as pd
# Torch
import torch
# Hugging Face
from transformers import AdamW, get_linear_schedule_with_warmup


from src.data.quickdata import get_train_dl, get_val_dl
from src.utils import nice_pbar, make_folder
from src.eval.metrics import kendall_tau
from src.models.baseline import MarkdownModel
from src.data import read_data


def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base')
    parser.add_argument('--train_mark_path', type=str, default='./data/train_mark.csv')
    parser.add_argument('--train_features_path', type=str, default='./data/train_fts.json')
    parser.add_argument('--val_mark_path', type=str, default='./data/val_mark.csv')
    parser.add_argument('--val_features_path', type=str, default='./data/val_fts.json')
    parser.add_argument('--val_path', type=str, default="./data/val.csv")

    parser.add_argument('--md_max_len', type=int, default=64)
    parser.add_argument('--total_max_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--n_workers', type=int, default=8)

    args = parser.parse_args()
    return args


def train(model, train_loader, val_loader, epochs):
    np.random.seed(0)
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
        preds = []
        labels = []

        tbar = nice_pbar(train_loader, len(train_loader), f"Train {epoch + 1}")

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

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_postfix(loss=avg_loss, lr=scheduler.get_last_lr())

        # TODO: Refactor to eval
        from src.eval import get_preds
        y_val, y_pred = get_preds(model, val_loader)
        val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
        val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred
        y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
        print("Preds score", kendall_tau(df_orders.loc[y_dummy.index], y_dummy))
        torch.save(model.state_dict(), "./outputs/model.bin")

    return model, y_pred


if __name__ == '__main__':
    args = parse_args()
    make_folder('./outputs')
    DATA_DIR = Path('../input')

    # TODO: Refactor to data
    val_df = pd.read_csv(args.val_path)
    order_df = pd.read_csv("../input/train_orders.csv").set_index("id")
    df_orders = pd.read_csv(
        DATA_DIR / 'train_orders.csv',
        index_col='id',
        squeeze=True,
    ).str.split()

    train_loader = get_train_dl(args)
    val_loader = get_val_dl(args)

    model = MarkdownModel(args.model_name_or_path)
    model = model.cuda()
    model, y_pred = train(model, train_loader, val_loader, epochs=args.epochs)
