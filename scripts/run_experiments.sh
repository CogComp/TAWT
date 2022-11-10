#!/usr/bin/env bash


# main experiments for (weighted) pre-training and (weighted) joint training
CUDA_VISIBLE_DEVICES=0 nohup sh scripts/run_single.sh > run_single_ner.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh scripts/run_joint_training_multi.sh > run_joint_training_multi_pos_chunking_predicate_ner.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh scripts/run_weighted_joint_training_multi.sh > run_weighted_joint_training_multi_pos_chunking_predicate_ner.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh scripts/run_pre_training_multi.sh > run_pre_training_multi_pos_chunking_predicate_ner.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh scripts/run_weighted_pre_training_multi.sh 1 > run_weighted_pre_training_multi_pos_chunking_predicate_ner_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh scripts/run_weighted_pre_training_multi.sh 10 > run_weighted_pre_training_multi_pos_chunking_predicate_ner_10.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh scripts/run_weighted_pre_training_multi.sh 30 > run_weighted_pre_training_multi_pos_chunking_predicate_ner_30.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup sh scripts/run_weighted_pre_training_multi.sh 100 > run_weighted_pre_training_multi_pos_chunking_predicate_ner_100.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup sh scripts/run_single.sh > run_single_predicate.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup sh scripts/run_joint_training_multi.sh > run_joint_training_multi_pos_chunking_ner_predicate.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh scripts/run_weighted_joint_training_multi.sh > run_weighted_joint_training_multi_pos_chunking_ner_predicate.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh scripts/run_pre_training_multi.sh > run_pre_training_multi_pos_chunking_ner_predicate.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh scripts/run_weighted_pre_training_multi.sh 1 > run_weighted_pre_training_multi_pos_chunking_ner_predicate_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh scripts/run_weighted_pre_training_multi.sh 10 > run_weighted_pre_training_multi_pos_chunking_ner_predicate_10.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh scripts/run_weighted_pre_training_multi.sh 30 > run_weighted_pre_training_multi_pos_chunking_ner_predicate_30.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup sh scripts/run_weighted_pre_training_multi.sh 100 > run_weighted_pre_training_multi_pos_chunking_ner_predicate_100.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup sh scripts/run_single.sh > run_single_chunking.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh scripts/run_joint_training_multi.sh > run_joint_training_multi_pos_predicate_ner_chunking.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh scripts/run_weighted_joint_training_multi.sh > run_weighted_joint_training_multi_pos_predicate_ner_chunking.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh scripts/run_pre_training_multi.sh > run_pre_training_multi_pos_predicate_ner_chunking.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh scripts/run_weighted_pre_training_multi.sh 1 > run_weighted_pre_training_multi_pos_predicate_ner_chunking_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh scripts/run_weighted_pre_training_multi.sh 10 > run_weighted_pre_training_multi_pos_predicate_ner_chunking_10.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh scripts/run_weighted_pre_training_multi.sh 30 > run_weighted_pre_training_multi_pos_predicate_ner_chunking_30.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup sh scripts/run_weighted_pre_training_multi.sh 100 > run_weighted_pre_training_multi_pos_predicate_ner_chunking_100.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup sh scripts/run_single.sh > run_single_pos.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup sh scripts/run_joint_training_multi.sh > run_joint_training_multi_chunking_predicate_ner_pos.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh scripts/run_weighted_joint_training_multi.sh > run_weighted_joint_training_multi_chunking_predicate_ner_pos.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh scripts/run_pre_training_multi.sh > run_pre_training_multi_chunking_predicate_ner_pos.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh scripts/run_weighted_pre_training_multi.sh 1 > run_weighted_pre_training_multi_chunking_predicate_ner_pos_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh scripts/run_weighted_pre_training_multi.sh 10 > run_weighted_pre_training_multi_chunking_predicate_ner_pos_10.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh scripts/run_weighted_pre_training_multi.sh 30 > run_weighted_pre_training_multi_chunking_predicate_ner_pos_30.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup sh scripts/run_weighted_pre_training_multi.sh 100 > run_weighted_pre_training_multi_chunking_predicate_ner_pos_100.log 2>&1 &


# main experiments for (weighted) normalized joint training
CUDA_VISIBLE_DEVICES=3 nohup sh scripts/run_joint_training_fixed_weights_multi.sh pos chunking predicate ner 0.05 0.05 0.05 0.85 500 > run_joint_training_fixed_weights_multi_pos_chunking_predicate_ner_normalized.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh scripts/run_joint_training_fixed_weights_multi.sh pos chunking ner predicate 0.03 0.03 0.03 0.91 300 > run_joint_training_fixed_weights_multi_pos_chunking_ner_predicate_normalized.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh scripts/run_joint_training_fixed_weights_multi.sh pos predicate ner chunking 0.01 0.01 0.01 0.97 100 > run_joint_training_fixed_weights_multi_pos_predicate_ner_chunking_normalized.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh scripts/run_joint_training_fixed_weights_multi.sh chunking predicate ner pos 0.01 0.01 0.01 0.97 100 > run_joint_training_fixed_weights_multi_chunking_predicate_ner_pos_normalized.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup sh scripts/run_weighted_joint_training_multi_normalized.sh pos chunking predicate ner 0.05 0.05 0.05 0.85 500 > run_weighted_joint_training_multi_pos_chunking_predicate_ner_normalized.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh scripts/run_weighted_joint_training_multi_normalized.sh pos chunking ner predicate 0.03 0.03 0.03 0.91 300 > run_weighted_joint_training_multi_pos_chunking_ner_predicate_normalized.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh scripts/run_weighted_joint_training_multi_normalized.sh pos predicate ner chunking 0.01 0.01 0.01 0.97 100 > run_weighted_joint_training_multi_pos_predicate_ner_chunking_normalized.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh scripts/run_weighted_joint_training_multi_normalized.sh chunking predicate ner pos 0.01 0.01 0.01 0.97 100 > run_weighted_joint_training_multi_chunking_predicate_ner_pos_normalized.log 2>&1 &

# final weights analysis
CUDA_VISIBLE_DEVICES=0 nohup sh scripts/run_joint_training_fixed_weights_multi.sh pos chunking predicate ner 0.04 0.04 0.05 0.87 500 > run_joint_training_fixed_weights_multi_pos_chunking_predicate_ner.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh scripts/run_joint_training_fixed_weights_multi.sh pos chunking ner predicate 0.04 0.04 0.05 0.87 300 > run_joint_training_fixed_weights_multi_pos_chunking_ner_predicate.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh scripts/run_joint_training_fixed_weights_multi.sh pos predicate ner chunking 0.04 0.04 0.05 0.87 100 > run_joint_training_fixed_weights_multi_pos_predicate_ner_chunking.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh scripts/run_joint_training_fixed_weights_multi.sh chunking predicate ner pos 0.04 0.04 0.03 0.89 100 > run_joint_training_fixed_weights_multi_chunking_predicate_ner_pos.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup sh scripts/run_pre_training_fixed_weights_multi.sh pos chunking predicate ner 0.92 0.05 0.04 500 > run_pre_training_fixed_weights_multi_pos_chunking_predicate_ner.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh scripts/run_pre_training_fixed_weights_multi.sh pos chunking ner predicate 0.33 0.35 0.32 300 > run_pre_training_fixed_weights_multi_pos_chunking_ner_predicate.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh scripts/run_pre_training_fixed_weights_multi.sh pos predicate ner chunking 0.32 0.30 0.37 100 > run_pre_training_fixed_weights_multi_pos_predicate_ner_chunking.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup sh scripts/run_pre_training_fixed_weights_multi.sh chunking predicate ner pos 0.68 0.01 0.31 100 > run_pre_training_fixed_weights_multi_chunking_predicate_ner_pos.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup sh scripts/run_joint_training_fixed_weights_multi.sh pos chunking predicate ner 0.003 0.003 0.003 0.990 500 > run_weighted_joint_training_fixed_weights_multi_pos_chunking_predicate_ner_normalized.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh scripts/run_joint_training_fixed_weights_multi.sh pos chunking ner predicate 0.002 0.002 0.002 0.995 300 > run_weighted_joint_training_fixed_weights_multi_pos_chunking_ner_predicate_normalized.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh scripts/run_joint_training_fixed_weights_multi.sh pos predicate ner chunking 0.0005 0.0005 0.0004 0.9987 100 > run_weighted_joint_training_fixed_weights_multi_pos_predicate_ner_chunking_normalized.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh scripts/run_joint_training_fixed_weights_multi.sh chunking predicate ner pos 0.0005 0.0004 0.0004 0.9986 100 > run_weighted_joint_training_fixed_weights_multi_chunking_predicate_ner_pos_normalized.log 2>&1 &


# factor analysis
CUDA_VISIBLE_DEVICES=0 nohup sh scripts/run_joint_training_multi_normalized_analysis.sh 500 1 > run_joint_training_multi_normalized_analysis_500_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh scripts/run_joint_training_multi_normalized_analysis.sh 500 2 > run_joint_training_multi_normalized_analysis_500_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh scripts/run_joint_training_multi_normalized_analysis.sh 500 4 > run_joint_training_multi_normalized_analysis_500_4.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh scripts/run_joint_training_multi_normalized_analysis.sh 500 8 > run_joint_training_multi_normalized_analysis_500_8.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh scripts/run_joint_training_multi_normalized_analysis.sh 500 16 > run_joint_training_multi_normalized_analysis_500_16.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh scripts/run_joint_training_multi_normalized_analysis.sh 500 32 > run_joint_training_multi_normalized_analysis_500_32.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh scripts/run_joint_training_multi_normalized_analysis.sh 500 64 > run_joint_training_multi_normalized_analysis_500_64.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup sh scripts/run_joint_training_multi_normalized_analysis.sh 500 128 > run_joint_training_multi_normalized_analysis_500_128.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup sh scripts/run_weighted_joint_training_multi_normalized_analysis.sh 500 1 > run_weighted_joint_training_multi_normalized_analysis_500_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh scripts/run_weighted_joint_training_multi_normalized_analysis.sh 500 2 > run_weighted_joint_training_multi_normalized_analysis_500_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh scripts/run_weighted_joint_training_multi_normalized_analysis.sh 500 4 > run_weighted_joint_training_multi_normalized_analysis_500_4.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh scripts/run_weighted_joint_training_multi_normalized_analysis.sh 500 8 > run_weighted_joint_training_multi_normalized_analysis_500_8.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh scripts/run_weighted_joint_training_multi_normalized_analysis.sh 500 16 > run_weighted_joint_training_multi_normalized_analysis_500_16.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh scripts/run_weighted_joint_training_multi_normalized_analysis.sh 500 32 > run_weighted_joint_training_multi_normalized_analysis_500_32.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh scripts/run_weighted_joint_training_multi_normalized_analysis.sh 500 64 > run_weighted_joint_training_multi_normalized_analysis_500_64.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup sh scripts/run_weighted_joint_training_multi_normalized_analysis.sh 500 128 > run_weighted_joint_training_multi_normalized_analysis_500_128.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup sh scripts/run_joint_training_multi_normalized_analysis.sh 500 10 > run_joint_training_multi_normalized_analysis_500_10.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh scripts/run_joint_training_multi_normalized_analysis.sh 1000 10 > run_joint_training_multi_normalized_analysis_1000_10.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh scripts/run_joint_training_multi_normalized_analysis.sh 2000 10 > run_joint_training_multi_normalized_analysis_2000_10.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh scripts/run_joint_training_multi_normalized_analysis.sh 4000 10 > run_joint_training_multi_normalized_analysis_4000_10.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh scripts/run_joint_training_multi_normalized_analysis.sh 8000 10 > run_joint_training_multi_normalized_analysis_8000_10.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup sh scripts/run_weighted_joint_training_multi_normalized_analysis.sh 500 10 > run_weighted_joint_training_multi_normalized_analysis_500_10.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup sh scripts/run_weighted_joint_training_multi_normalized_analysis.sh 1000 10 > run_weighted_joint_training_multi_normalized_analysis_1000_10.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh scripts/run_weighted_joint_training_multi_normalized_analysis.sh 2000 10 > run_weighted_joint_training_multi_normalized_analysis_2000_10.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh scripts/run_weighted_joint_training_multi_normalized_analysis.sh 4000 10 > run_weighted_joint_training_multi_normalized_analysis_4000_10.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh scripts/run_weighted_joint_training_multi_normalized_analysis.sh 8000 10 > run_weighted_joint_training_multi_normalized_analysis_8000_10.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup sh scripts/run_joint_training_multi_normalized_analysis.sh 1000 8 > run_joint_training_multi_normalized_analysis_1000_8.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup sh scripts/run_joint_training_multi_normalized_analysis.sh 2000 8 > run_joint_training_multi_normalized_analysis_2000_8.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh scripts/run_joint_training_multi_normalized_analysis.sh 4000 8 > run_joint_training_multi_normalized_analysis_4000_8.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh scripts/run_joint_training_multi_normalized_analysis.sh 8000 8 > run_joint_training_multi_normalized_analysis_8000_8.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup sh scripts/run_weighted_joint_training_multi_normalized_analysis.sh 1000 8 > run_weighted_joint_training_multi_normalized_analysis_1000_8.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh scripts/run_weighted_joint_training_multi_normalized_analysis.sh 2000 8 > run_weighted_joint_training_multi_normalized_analysis_2000_8.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup sh scripts/run_weighted_joint_training_multi_normalized_analysis.sh 4000 8 > run_weighted_joint_training_multi_normalized_analysis_4000_8.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh scripts/run_weighted_joint_training_multi_normalized_analysis.sh 8000 8 > run_weighted_joint_training_multi_normalized_analysis_8000_8.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup sh scripts/run_single_analysis.sh 500 > run_single_ner_analysis_500.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh scripts/run_single_analysis.sh 1000 > run_single_ner_analysis_1000.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh scripts/run_single_analysis.sh 2000 > run_single_ner_analysis_2000.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh scripts/run_single_analysis.sh 4000 > run_single_ner_analysis_4000.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh scripts/run_single_analysis.sh 8000 > run_single_ner_analysis_8000.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup sh scripts/run_single_ner_analysis.sh 1500 > run_single_ner_analysis_1500.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh scripts/run_single_ner_analysis.sh 2500 > run_single_ner_analysis_2500.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh scripts/run_single_ner_analysis.sh 4500 > run_single_ner_analysis_4500.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh scripts/run_single_ner_analysis.sh 8500 > run_single_ner_analysis_8500.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh scripts/run_single_ner_analysis.sh 16500 > run_single_ner_analysis_16500.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh scripts/run_single_ner_analysis.sh 32500 > run_single_ner_analysis_32500.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh scripts/run_single_ner_analysis.sh 64500 > run_single_ner_analysis_64500.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup sh scripts/run_single_ner_analysis.sh 17000 > run_single_ner_analysis_17000.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh scripts/run_single_ner_analysis.sh 34000 > run_single_ner_analysis_34000.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh scripts/run_single_ner_analysis.sh 68000 > run_single_ner_analysis_68000.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh scripts/run_single_ner_analysis.sh 10500 > run_single_ner_analysis_10500.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh scripts/run_single_ner_analysis.sh 21000 > run_single_ner_analysis_21000.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh scripts/run_single_ner_analysis.sh 42000 > run_single_ner_analysis_42000.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup sh scripts/run_single_ner_analysis.sh 84000 > run_single_ner_analysis_84000.log 2>&1 &