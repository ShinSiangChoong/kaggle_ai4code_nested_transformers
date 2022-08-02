#!/usr/bin/sh

python src/training/train.py \
--max_n_cells 126 \
--max_len 64 \
--batch_size 4 \
--accumulation_steps 4 \
--epochs 30 \
--n_workers 1 \
--wandb_mode online