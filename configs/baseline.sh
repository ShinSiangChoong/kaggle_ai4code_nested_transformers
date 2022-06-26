#!/usr/bin/sh

python src/training/train.py \
--max_n_codes 20 \
--max_md_len 64 \
--max_len 512 \
--batch_size 4 \
--accumulation_steps 8 \
--epochs 5 \
--n_workers 8 \
--wandb_mode online
