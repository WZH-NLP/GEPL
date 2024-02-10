#!/bin/bash
#SBATCH -e ./results/Cross-domain/new/conll_literature_max_no_context_wandb.err
#SBATCH -o ./results/Cross-domain/new/conll_literature_max_no_context_wandb.out
#SBATCH -J fewNER_literature
#SBATCH --gres=gpu:1
#SBATCH --time=999:00:00
#source /data/wanghanbing-slurm/.bashrc
conda activate ent


DATASET=$1
SHOTS=$2
PROMPT=$3
TEMPLATE=$4
TRAIN_SEED=$5
SAMPLE_SEED=$6
CHECK_POINT=$7

SEEDED_SUFFIX="${SHOTS}_${SAMPLE_SEED}"
MODEL_NAME="model_da_${PROMPT}_${TEMPLATE}_${SEEDED_SUFFIX}_${TRAIN_SEED}"

python3 transformers_continual_trainer.py \
  --dataset $DATASET \
  --data_dir dataset/$DATASET \
  --checkpoint $CHECK_POINT \
  --learning_rate 5e-5 \
  --batch_size 4 \
  --dropout 0.2 \
  --model_folder models/$DATASET/$MODEL_NAME \
  --device cuda \
  --prompt $PROMPT \
  --template $TEMPLATE \
  --search_pool target \
  --percent_filename_suffix $SEEDED_SUFFIX \
  --seed $TRAIN_SEED