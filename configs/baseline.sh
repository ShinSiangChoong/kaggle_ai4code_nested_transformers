#!/usr/bin/sh

python src/training/train.py \
--max_n_cells 254 \
--max_len 64 \
--batch_size 6 \
--accumulation_steps 4 \
--epochs 5 \
--n_workers 1 \
# --wandb_mode online