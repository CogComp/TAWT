#!/usr/bin/env bash

echo "Settings $1 $2 $3 $4 $5 $6 $7 $8 "

export WORKING_DIR=/path/to/working/dir
export MAX_LENGTH=128
export BERT_MODEL=bert-base-cased
export BATCH_SIZE=32
export NUM_EPOCHS=4
export SEED=1

python run_pre_training_fixed_weights_multi.py \
--data_dir $WORKING_DIR/data \
--model_type bert \
--auxiliary_task_list $1 $2 $3 \
--main_task $4 \
--task_weights $5 $6 $7 \
--main_task_train_sent_num $8 \
--model_name_or_path $BERT_MODEL \
--output_dir $WORKING_DIR/bert-base-cased-pre-training-fixed-weights-multi-$1-$2-$3-$4 \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--per_gpu_eval_batch_size $BATCH_SIZE \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
