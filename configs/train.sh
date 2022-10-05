#!/usr/bin/sh

python src/training/train.py \
--model_name_or_path 'microsoft/codebert-base' \
--tokenizer_name_or_path 'microsoft/codebert-base' \
--max_n_cells 126 \
--max_len 64 \
--batch_size 2 \
--accumulation_steps 8 \
--epochs 30 \
--n_workers 1 \
--wandb_mode disabled \
--wandb_name 'codebert' \
--ellipses_token_id 734 \
--emb_size 768 \
--output_dir './outputs'