# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """


import argparse
import glob
import logging
import os
import random
import sklearn

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from modeling_multi_bert import MTDNN
from utils_cross_task import convert_examples_to_features, get_labels, read_examples_from_file


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), ())

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode, task):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            mode, task, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length)
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args.data_dir, mode, task)
        features = convert_examples_to_features(
            examples,
            labels,
            args.max_seq_length,
            tokenizer,
            auxiliary_task_list=args.auxiliary_task_list,
            main_task=args.main_task,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=pad_token_label_id,
        )
        # if args.local_rank in [-1, 0]:
        #     logger.info("Saving features into cached file %s", cached_features_file)
        #     torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    if task == args.main_task and mode == 'train':
        features = features[:args.main_task_train_sent_num]
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_task_id = torch.tensor([f.task_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_task_id)
    return dataset


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, task, prefix=""):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode, task=task)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running %s evaluation %s *****", task, prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "task_id": batch[4]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    if task == 'ner' or task == 'chunking':
        results = {
            "loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }
    elif task == 'predicate':
        y_pred = []
        y_label = []
        for preds in preds_list:
            y_pred.extend(preds)
        for labels in out_label_list:
            y_label.extend(labels)
        results = {
            "loss": eval_loss,
            "accuracy": sklearn.metrics.accuracy_score(y_label, y_pred),
            "f1": sklearn.metrics.f1_score(y_label, y_pred, pos_label='predicate'),
        }
    else:
        y_pred = []
        y_label = []
        for preds in preds_list:
            y_pred.extend(preds)
        for labels in out_label_list:
            y_label.extend(labels)
        results = {
            "loss": eval_loss,
            "accuracy": sklearn.metrics.accuracy_score(y_label, y_pred),
            "micro f1": sklearn.metrics.f1_score(y_label, y_pred, average='micro'),
            "macro f1": sklearn.metrics.f1_score(y_label, y_pred, average='macro'),
        }

    logger.info("***** Eval %s results %s *****", task, prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


def main_train(args, train_main_dataset, model, tokenizer, main_labels, pad_token_label_id,  main_task):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    main_train_sampler = RandomSampler(train_main_dataset) \
        if args.local_rank == -1 else DistributedSampler(train_main_dataset)
    main_train_dataloader = DataLoader(train_main_dataset, sampler=main_train_sampler, batch_size=args.train_batch_size)

    train_dataloader = []
    for i, main_batch in enumerate(main_train_dataloader):
        train_dataloader.append(main_batch)
    random.shuffle(train_dataloader)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num %s examples = %d", main_task, len(train_main_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        random.shuffle(train_dataloader)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "task_id": batch[4]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well

                        main_results, _ = evaluate(args, model, tokenizer, main_labels, pad_token_label_id, mode="dev",
                                                   task=main_task)
                        for key, value in main_results.items():
                            tb_writer.add_scalar("{}_eval_{}".format(main_task, key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def auxiliary_train(args, train_auxiliary_dataset_list, model, tokenizer, auxiliary_labels_list,
                    pad_token_label_id, auxiliary_task_list):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    auxiliary_train_dataloader_list = []
    for train_auxiliary_dataset in train_auxiliary_dataset_list:
        auxiliary_train_sampler = RandomSampler(train_auxiliary_dataset) \
            if args.local_rank == -1 else DistributedSampler(train_auxiliary_dataset)
        auxiliary_train_dataloader = DataLoader(train_auxiliary_dataset, sampler=auxiliary_train_sampler,
                                                batch_size=args.train_batch_size)
        auxiliary_train_dataloader_list.append(auxiliary_train_dataloader)

    train_dataloader = []
    for auxiliary_train_dataloader in auxiliary_train_dataloader_list:
        for i, auxiliary_batch in enumerate(auxiliary_train_dataloader):
            train_dataloader.append(auxiliary_batch)
    random.shuffle(train_dataloader)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    for x in range(len(auxiliary_task_list)):
        logger.info("  Num %s examples = %d", auxiliary_task_list[x], len(train_auxiliary_dataset_list[x]))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    task_weights = [float(weight) for weight in args.task_weights]
    print('task weights', task_weights)
    for _ in train_iterator:
        random.shuffle(train_dataloader)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "task_id": batch[4]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # add weighted loss
            task_id = int(inputs['task_id'][0])
            loss = loss * task_weights[task_id] * len(auxiliary_task_list)

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        auxiliary_results_list = []
                        for x in range(len(auxiliary_task_list)):
                            auxiliary_results, _ = evaluate(args, model, tokenizer, auxiliary_labels_list[x],
                                                            pad_token_label_id, mode="dev", task=auxiliary_task_list[x])
                            auxiliary_results_list.append(auxiliary_results)

                        for x in range(len(auxiliary_task_list)):
                            auxiliary_results = auxiliary_results_list[x]
                            for key, value in auxiliary_results.items():
                                tb_writer.add_scalar("{}_eval_{}".format(auxiliary_task_list[x], key), value,
                                                     global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "auxiliary-checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def train(args, train_auxiliary_dataset_list, train_main_dataset, model, tokenizer, auxiliary_labels_list, main_labels,
          pad_token_label_id, auxiliary_task_list, main_task):
    """ Train the model """
    if args.pre_train_only:
        # pre-training on auxiliary datasets
        print('pre-training on auxiliary tasks')
        return auxiliary_train(args, train_auxiliary_dataset_list, model, tokenizer, auxiliary_labels_list,
                               pad_token_label_id, auxiliary_task_list)
    else:
        # pre-training on auxiliary datasets
        print('pre-training on auxiliary tasks')
        auxiliary_train(args, train_auxiliary_dataset_list, model, tokenizer, auxiliary_labels_list,
                        pad_token_label_id, auxiliary_task_list)
        # fine-tuning on the main dataset
        print('fine-tunning on the main task')
        return main_train(args, train_main_dataset, model, tokenizer, main_labels, pad_token_label_id, main_task)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--auxiliary_task_list",
        default=[],
        nargs='+',
        required=True,
        help="The auxiliary task list."
    )

    parser.add_argument(
        "--main_task",
        default=None,
        type=str,
        required=True,
        help="The main task.",
    )
    parser.add_argument("--pre_train_only", action='store_true', default=False,
                        help="Only pre-train on auxiliary datasets")

    parser.add_argument(
        "--task_weights",
        default=[],
        nargs='+',
        required=True,
        help="Task weights."
    )

    parser.add_argument(
        "--main_task_train_sent_num",
        default=0,
        type=int,
        help="The number of training sentences in the main task"
    )

    # Other parameters

    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--keep_accents", action="store_const", const=True, help="Set this flag if model is trained with accents."
    )
    parser.add_argument(
        "--strip_accents", action="store_const", const=True, help="Set this flag if model is trained without accents."
    )
    parser.add_argument("--use_fast", action="store_const", const=True, help="Set this flag to use fast tokenization.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare auxiliary tasks and main task
    auxiliary_labels_list = []
    auxiliary_num_labels_list = []
    for x in range(len(args.auxiliary_task_list)):
        auxiliary_labels = get_labels(args.data_dir + '/{}_labels.txt'.format(args.auxiliary_task_list[x]))
        auxiliary_labels_list.append(auxiliary_labels)
        auxiliary_num_labels_list.append(len(auxiliary_labels))
    main_labels = get_labels(args.data_dir + '/{}_labels.txt'.format(args.main_task))
    main_num_labels = len(main_labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    auxiliary_id2label_list = []
    auxiliary_label2id_list = []
    for x in range(len(args.auxiliary_task_list)):
        auxiliary_id2label = {str(i): label for i, label in enumerate(auxiliary_labels_list[x])}
        auxiliary_label2id = {label: i for i, label in enumerate(auxiliary_labels_list[x])}
        auxiliary_id2label_list.append(auxiliary_id2label)
        auxiliary_label2id_list.append(auxiliary_label2id)
    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        auxiliary_id2label_list=auxiliary_id2label_list,
        auxiliary_label2id_list=auxiliary_label2id_list,
        main_id2label={str(i): label for i, label in enumerate(main_labels)},
        main_label2id={label: i for i, label in enumerate(main_labels)},
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer_args = {k: v for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS}
    logger.info("Tokenizer arguments: %s", tokenizer_args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        **tokenizer_args,
    )
    model = MTDNN.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        auxiliary_num_labels_list=auxiliary_num_labels_list,
        main_num_labels=main_num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_auxiliary_dataset_list = []
        for x in range(len(args.auxiliary_task_list)):
            train_auxiliary_dataset = load_and_cache_examples(args, tokenizer, auxiliary_labels_list[x],
                                                              pad_token_label_id, mode="train",
                                                              task=args.auxiliary_task_list[x])
            train_auxiliary_dataset_list.append(train_auxiliary_dataset)

        train_main_dataset = load_and_cache_examples(args, tokenizer, main_labels, pad_token_label_id, mode="train",
                                                     task=args.main_task)
        global_step, tr_loss = train(args, train_auxiliary_dataset_list, train_main_dataset, model, tokenizer,
                                     auxiliary_labels_list, main_labels, pad_token_label_id,
                                     args.auxiliary_task_list, args.main_task)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    auxiliary_results_list = [{} for i in range(len(args.auxiliary_task_list))]
    main_results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, **tokenizer_args)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = MTDNN.from_pretrained(checkpoint,
                                          auxiliary_num_labels_list=auxiliary_num_labels_list,
                                          main_num_labels=main_num_labels)
            model.to(args.device)
            auxiliary_result_list = []
            for x in range(len(args.auxiliary_task_list)):
                auxiliary_result, _ = evaluate(args, model, tokenizer, auxiliary_labels_list[x], pad_token_label_id,
                                               mode="dev", task=args.auxiliary_task_list[x], prefix=global_step)
                auxiliary_result_list.append(auxiliary_result)
            main_result, _ = evaluate(args, model, tokenizer, main_labels, pad_token_label_id,
                                      mode="dev", task=args.main_task, prefix=global_step)
            if global_step:
                new_auxiliary_result_list = []
                for auxiliary_result in auxiliary_result_list:
                    auxiliary_result = {"{}_{}".format(global_step, k): v for k, v in auxiliary_result.items()}
                    new_auxiliary_result_list.append(auxiliary_result)
                auxiliary_result_list = new_auxiliary_result_list
                main_result = {"{}_{}".format(global_step, k): v for k, v in main_result.items()}
            for x in range(len(args.auxiliary_task_list)):
                auxiliary_results = auxiliary_results_list[x]
                auxiliary_result = auxiliary_result_list[x]
                auxiliary_results.update(auxiliary_result)
            main_results.update(main_result)
        auxiliary_output_eval_file_list = []
        for x in range(len(args.auxiliary_task_list)):
            auxiliary_output_eval_file_list.append(os.path.join(args.output_dir,
                                                                "{}_eval_results.txt".format(args.auxiliary_task_list[x])))
        main_output_eval_file = os.path.join(args.output_dir, "{}_eval_results.txt".format(args.main_task))
        for x in range(len(args.auxiliary_task_list)):
            auxiliary_output_eval_file = auxiliary_output_eval_file_list[x]
            auxiliary_results = auxiliary_results_list[x]
            with open(auxiliary_output_eval_file, "w") as writer:
                for key in sorted(auxiliary_results.keys()):
                    writer.write("{} = {}\n".format(key, str(auxiliary_results[key])))
        with open(main_output_eval_file, "w") as writer:
            for key in sorted(main_results.keys()):
                writer.write("{} = {}\n".format(key, str(main_results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, **tokenizer_args)
        model = MTDNN.from_pretrained(args.output_dir,
                                      auxiliary_num_labels_list=auxiliary_num_labels_list,
                                      main_num_labels=main_num_labels)
        model.to(args.device)
        auxiliary_result_list = []
        auxiliary_predictions_list = []
        for x in range(len(args.auxiliary_task_list)):
            auxiliary_result, auxiliary_predictions = evaluate(args, model, tokenizer, auxiliary_labels_list[x],
                                                               pad_token_label_id, mode="test",
                                                               task=args.auxiliary_task_list[x])
            auxiliary_result_list.append(auxiliary_result)
            auxiliary_predictions_list.append(auxiliary_predictions)
        main_result, main_predictions = evaluate(args, model, tokenizer, main_labels, pad_token_label_id, mode="test",
                                                 task=args.main_task)
        # Save results
        auxiliary_output_test_results_file_list = []
        for x in range(len(args.auxiliary_task_list)):
            auxiliary_output_test_results_file = os.path.join(args.output_dir,
                                                              "{}_test_results.txt".format(args.auxiliary_task_list[x]))
            auxiliary_output_test_results_file_list.append(auxiliary_output_test_results_file)
        main_output_test_results_file = os.path.join(args.output_dir, "{}_test_results.txt".format(args.main_task))
        for x in range(len(args.auxiliary_task_list)):
            auxiliary_output_test_results_file = auxiliary_output_test_results_file_list[x]
            auxiliary_result = auxiliary_result_list[x]
            with open(auxiliary_output_test_results_file, "w") as writer:
                for key in sorted(auxiliary_result.keys()):
                    writer.write("{} = {}\n".format(key, str(auxiliary_result[key])))
        with open(main_output_test_results_file, "w") as writer:
            for key in sorted(main_result.keys()):
                writer.write("{} = {}\n".format(key, str(main_result[key])))
        # Save predictions
        auxiliary_output_test_predictions_file_list = []
        for x in range(len(args.auxiliary_task_list)):
            auxiliary_output_test_predictions_file = os.path.join(args.output_dir,
                                                                  "{}_test_predictions.txt".format(args.auxiliary_task_list[x]))
            auxiliary_output_test_predictions_file_list.append(auxiliary_output_test_predictions_file)
        main_output_test_predictions_file = os.path.join(args.output_dir, "{}_test_predictions.txt".format(args.main_task))
        for x in range(len(args.auxiliary_task_list)):
            auxiliary_output_test_predictions_file = auxiliary_output_test_predictions_file_list[x]
            auxiliary_predictions = auxiliary_predictions_list[x]
            with open(auxiliary_output_test_predictions_file, "w") as writer:
                with open(os.path.join(args.data_dir, "test_{}.txt".format(args.auxiliary_task_list[x])), "r") as f:
                    example_id = 0
                    for line in f:
                        if example_id == len(auxiliary_predictions):
                            break
                        if line.startswith("-DOCSTART-") or line == "" or line == "\n" or len(line.strip()) == 0:
                            writer.write(line)
                            if not auxiliary_predictions[example_id]:
                                example_id += 1
                        elif auxiliary_predictions[example_id]:
                            output_line = line.split()[0] + " " + auxiliary_predictions[example_id].pop(0) + "\n"
                            writer.write(output_line)
                        else:
                            logger.warning("Maximum sequence length exceeded: No pos prediction for '%s'.",
                                           line.split()[0])
        with open(main_output_test_predictions_file, "w") as writer:
            with open(os.path.join(args.data_dir, "test_{}.txt".format(args.main_task)), "r") as f:
                example_id = 0
                for line in f:
                    if example_id == len(main_predictions):
                        break
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n" or len(line.strip()) == 0:
                        writer.write(line)
                        if not main_predictions[example_id]:
                            example_id += 1
                    elif main_predictions[example_id]:
                        output_line = line.split()[0] + " " + main_predictions[example_id].pop(0) + "\n"
                        writer.write(output_line)
                    else:
                        logger.warning("Maximum sequence length exceeded: No ner prediction for '%s'.", line.split()[0])


if __name__ == "__main__":
    main()

