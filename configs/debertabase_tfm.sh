#!/usr/bin/sh

python src/training/train.py \
--model_name_or_path 'microsoft/deberta-v3-base' \
--tokenizer_name_or_path 'microsoft/deberta-v3-base' \
--pretrained_mlm_path './outputs/debertav3base_trained_mlm_with_ext_data/model-2.bin' \
--max_n_cells 126 \
--max_len 64 \
--batch_size 2 \
--accumulation_steps 8 \
--epochs 30 \
--n_workers 1 \
--wandb_mode online \
--wandb_name 'Deberta-v3-base 126 MLM' \
--ellipses_token_id 2 \
--emb_size 768 \
--output_dir './outputs/debertav3base_nested_tfm_126'