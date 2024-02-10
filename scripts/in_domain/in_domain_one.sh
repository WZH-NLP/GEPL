#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --job-name=in_domain
#SBATCH --gres=gpu:1 
#SBATCH --exclude ink-ellie,ink-molly,ink-ron

DATASET=$1
SHOTS=$2
PROMPT=$3
TEMPLATE=$4
TRAIN_SEED=$5
SAMPLE_SEED=$6


SEEDED_SUFFIX="${SHOTS}_${SAMPLE_SEED}"
MODEL_NAME="model_${PROMPT}_${TEMPLATE}_${SEEDED_SUFFIX}_${TRAIN_SEED}"


python3 transformers_trainer.py \
  --dataset $DATASET \
  --data_dir dataset/$DATASET \
  --learning_rate 2e-5 \
  --batch_size 4 \
  --dropout 0.5 \
  --model_folder models/$DATASET/$MODEL_NAME \
  --device cuda \
  --percent_filename_suffix $SEEDED_SUFFIX \
  --prompt $PROMPT \
  --template $TEMPLATE \
  --num_epochs 50 \
  --max_no_incre 30 \
  --seed $TRAIN_SEED