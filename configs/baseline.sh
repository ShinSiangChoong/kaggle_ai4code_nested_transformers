#!/usr/bin/sh

python src/training/train.py \
--model_name_or_path 'microsoft/codebert-base' \
--tokenizer_name_or_path './outputs/tokenizer_codebert_ext_data' \
--pretrained_mlm_path './outputs/model-5.bin' \
--max_n_cells 254 \
--max_len 64 \
--batch_size 2 \
--accumulation_steps 8 \
--epochs 30 \
--n_workers 1 \
--wandb_mode online \
--output_dir './outputs/codebert_mlm_tfm'