#!/bin/bash
#SBATCH -e ./results/max_no_context_200/Conll_literature_train_0.4_2.err
#SBATCH -o ./results/max_no_context_200/Conll_literature_train_0.4_2.out
#SBATCH -J MTD_conll
#SBATCH --nodelist=gpu07
#SBATCH --gres=gpu:1
#SBATCH --time=999:00:00


GPUID=$1
echo "Run on GPU $GPUID"
#TRAIN=$3   0.5Graph_3Target_0.05mse_pretrain.out
#TEST=$4
# data
DATASET=$2
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
DATA_ROOT=$PROJECT_ROOT/dataset/

# model
TOKENIZER_TYPE=bert
SPAN_TYPE=bert
TYPE_TYPE=bert

TOKENIZER_NAME='./cached_models/bert-base-cased'
SPAN_MODEL_NAME='./cached_models/bert-base-cased'
TYPE_MODEL_NAME='./cached_models/bert-base-cased'
# TOKENIZER_NAME='./cached_models/literature_train_0.4/checkpoint-best-type-test'
# SPAN_MODEL_NAME='./cached_models/literature_train_0.4/checkpoint-best-span-test'
# TYPE_MODEL_NAME='./cached_models/literature_train_0.4/checkpoint-best-type-test'

TAU_SPAN=$3
TAU_TYPE=$4

# params
LR=1e-5
WEIGHT_DECAY=1e-4
EPOCH=50
SEED=0

ADAM_EPS=1e-8
ADAM_BETA1=0.9
ADAM_BETA2=0.98
WARMUP=500

TRAIN_BATCH_SRC=16
TRAIN_BATCH=16
EVAL_BATCH=32

MU=$5
ALPHA=0.3
BETA=0.7
EPS_SPAN=0.5
EPS_TYPE=0.0
L=3

# output CUDA_VISIBLE_DEVICES=3 # 0.85 0.7 0.55 0.4 0.25 0.1
OUTPUT=$PROJECT_ROOT/ptms/$DATASET/train_0.4_2

CUDA_DEVICE_ORDER=PCI_BUS_ID python3 -u run_script.py --data_dir $DATA_ROOT \
  --span_model_name_or_path $SPAN_MODEL_NAME \
  --type_model_name_or_path $TYPE_MODEL_NAME \
  --proportion 0.4 \
  --prompt_or_not \
  --prompt max \
  --template no_context \
  --search_pool respective\
  --output_dir $OUTPUT \
  --tokenizer_name_or_path $TOKENIZER_NAME \
  --cache_dir $PROJECT_ROOT/cached_models \
  --max_seq_length 200 \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --adam_epsilon $ADAM_EPS \
  --adam_beta1 $ADAM_BETA1 \
  --adam_beta2 $ADAM_BETA2 \
  --max_grad_norm 1.0 \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_gpu_train_batch_size_src $TRAIN_BATCH_SRC \
  --per_gpu_train_batch_size $TRAIN_BATCH \
  --per_gpu_eval_batch_size $EVAL_BATCH \
  --gradient_accumulation_steps 1 \
  --logging_steps 100 \
  --save_steps 100000 \
  --do_train True \
  --evaluate_during_training \
  --seed $SEED \
  --overwrite_output_dir \
  --model_type $TOKENIZER_TYPE \
  --dataset $DATASET \
  --src_dataset conll2003 \
  --tau_span $TAU_SPAN \
  --tau_type $TAU_TYPE \
  --mu $MU \
  --alpha $ALPHA \
  --beta $BETA \
  --eps_span $EPS_SPAN \
  --eps_type $EPS_TYPE \
  --L $L \
