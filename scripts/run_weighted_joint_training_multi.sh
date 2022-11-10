#!/usr/bin/env bash

export WORKING_DIR=/path/to/working/dir
export MAX_LENGTH=128
export BERT_MODEL=bert-base-cased
export BATCH_SIZE=32
export NUM_EPOCHS=4
export SEED=1

python code/run_weighted_joint_training_multi.py \
--data_dir $WORKING_DIR/data \
--model_type bert \
--auxiliary_task_list pos chunking ner \
--main_task predicate \
--main_task_train_sent_num 300 \
--model_name_or_path $BERT_MODEL \
--gradient_sampling_batch_size 64 \
--normalized_gradients \
--output_dir $WORKING_DIR/bert-base-cased-weighted-joint-training-multi-pos-chunking-ner-predicate \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--per_gpu_eval_batch_size $BATCH_SIZE \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
