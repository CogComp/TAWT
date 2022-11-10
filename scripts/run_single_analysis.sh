#!/usr/bin/env bash

echo "Train sent num: $1"

export WORKING_DIR=/path/to/working/dir
export MAX_LENGTH=128
export BERT_MODEL=bert-base-cased
export BATCH_SIZE=32
export NUM_EPOCHS=4
export SEED=1

python code/run_single_analysis.py \
--data_dir $WORKING_DIR/data \
--model_type bert \
--task ner \
--train_sent_num $1 \
--labels $WORKING_DIR/data/ner_labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $WORKING_DIR/bert-base-cased-single-ner-$1 \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--per_gpu_eval_batch_size $BATCH_SIZE \
--seed $SEED \
--do_train \
--do_eval \
--do_predict