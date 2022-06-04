#!/usr/bin/sh

python src/training/train.py \
--md_max_len 64 \
--total_max_len 512 \
--batch_size 4 \
--accumulation_steps 8 \
--epochs 5 \
--n_workers 8