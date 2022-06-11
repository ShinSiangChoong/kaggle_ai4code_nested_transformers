#!/bin/bash
#
#SBATCH --partition=gpu-titan
#SBATCH --job-name=Meow
#SBATCH --output=/home/user/jackmin/slurm_logs/%j.out
#SBATCH --error=/home/user/jackmin/slurm_logs/%j.err
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --qos=long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wid190507@siswa.um.edu.my

BASE_DIR=$1
if [[ ! $BASE_DIR ]]; then
  echo 'Please specify a base directory'
  exit 1
fi
CONDA_ENV=$2
if [[ ! $CONDA_ENV ]]; then
  echo 'Please specify a conda env to use'
  exit 1
fi

echo ======================================
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV
cd $BASE_DIR
echo ======================================
pwd
which python
echo ======================================
nvidia-smi
echo ======================================
env
echo ======================================
source env.sh
python src/training/train.py \
--md_max_len 64 \
--total_max_len 512 \
--batch_size 16 \
--accumulation_steps 2 \
--epochs 5 \
--n_workers 8 \
--wandb_mode online \
--output_dir 10pct