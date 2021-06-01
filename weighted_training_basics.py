import copy
import numpy as np
from torch.autograd import grad


def get_flatten_vectors(gradients_tensors):
    flatten_vectors = []
    for gradient_parts in gradients_tensors:
        if gradient_parts is not None:
            flatten_vectors.extend(list(gradient_parts.view(-1).cpu().data.numpy()))
    return flatten_vectors


def get_average_feature_gradients(static_model, dataloader, args):
    model = copy.deepcopy(static_model)
    for param in model.bert.parameters():
        param.requires_grad = True
    model.eval()
    eval_loss = 0.0
    nb_eval_steps = 0
    for i, batch in enumerate(dataloader):
        if i == 1:
            break
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "task_id": batch[4]}
        if args.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use segment_ids
        outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    # print('eval loss', eval_loss)
    feature_gradients = grad(eval_loss, model.bert.parameters(), retain_graph=False, create_graph=False,
                             allow_unused=True)
    # print('feature gradients', feature_gradients)
    return feature_gradients


def get_task_weights_gradients(model, pos_train_dataloader, ner_train_dataloader, args):
    source_average_feature_gradients = get_average_feature_gradients(model, pos_train_dataloader, args)
    target_average_feature_gradients = get_average_feature_gradients(model, ner_train_dataloader, args)
    source_average_feature_gradients = np.array(get_flatten_vectors(list(source_average_feature_gradients)))
    target_average_feature_gradients = np.array(get_flatten_vectors(list(target_average_feature_gradients)))
    left_operator = target_average_feature_gradients
    task_weights_gradients = np.array([-np.dot(left_operator, source_average_feature_gradients),
                                       -np.dot(left_operator, target_average_feature_gradients)])
    if args.normalized_gradients:
        task_weights_gradients /= np.abs(task_weights_gradients[1])
        print('normalized task weights gradients', task_weights_gradients)
    else:
        if args.inv_constant == 0.0:
            args.inv_constant = 1 / np.abs(task_weights_gradients[1])
        task_weights_gradients *= args.inv_constant
        print('inv constant', args.inv_constant)
        print('unnormalized task weights gradients', task_weights_gradients)
    return task_weights_gradients


def get_task_weights_gradients_multi(model, auxiliary_train_dataloader_list, main_train_dataloader, args):
    source_average_feature_gradients_list = []
    for x in range(len(args.auxiliary_task_list)):
        source_average_feature_gradients = get_average_feature_gradients(model, auxiliary_train_dataloader_list[x],
                                                                         args)
        source_average_feature_gradients = np.array(get_flatten_vectors(list(source_average_feature_gradients)))
        source_average_feature_gradients_list.append(source_average_feature_gradients)
    target_average_feature_gradients = get_average_feature_gradients(model, main_train_dataloader, args)
    target_average_feature_gradients = np.array(get_flatten_vectors(list(target_average_feature_gradients)))
    left_operator = target_average_feature_gradients
    task_weights_gradients = []
    for x in range(len(args.auxiliary_task_list)):
        source_average_feature_gradients = source_average_feature_gradients_list[x]
        # task_weights_gradients.append(-np.dot(left_operator, source_average_feature_gradients))
        task_weights_gradients.append(-get_cosine(left_operator, source_average_feature_gradients))
    # task_weights_gradients.append(-np.dot(left_operator, target_average_feature_gradients))
    task_weights_gradients.append(-get_cosine(left_operator, target_average_feature_gradients))
    task_weights_gradients = np.array(task_weights_gradients)
    print('task weights gradients', task_weights_gradients)
    if args.normalized_gradients:
        # task_weights_gradients /= np.abs(task_weights_gradients[-1])
        # task_weights_gradients *= 1
        # max_interval = np.max(task_weights_gradients) - np.min(task_weights_gradients)
        # print('max interval', max_interval)
        # task_weights_gradients /= max_interval
        print('normalized task weights gradients', task_weights_gradients)
    else:
        # if args.inv_constant == 0.0:
            # args.inv_constant = 1 / np.abs(task_weights_gradients[-1])
        # args.inv_constant = 1
        task_weights_gradients *= args.inv_constant
        print('inv constant', args.inv_constant)
        print('unnormalized task weights gradients', task_weights_gradients)
    return task_weights_gradients


def get_cosine(a, b):
    return np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))


def get_task_weights_gradients_multi_pre(model, auxiliary_train_dataloader_list, main_train_dataloader, args):
    source_average_feature_gradients_list = []
    for x in range(len(args.auxiliary_task_list)):
        source_average_feature_gradients = get_average_feature_gradients(model, auxiliary_train_dataloader_list[x],
                                                                         args)
        source_average_feature_gradients = np.array(get_flatten_vectors(list(source_average_feature_gradients)))
        source_average_feature_gradients_list.append(source_average_feature_gradients)
    target_average_feature_gradients = get_average_feature_gradients(model, main_train_dataloader, args)
    target_average_feature_gradients = np.array(get_flatten_vectors(list(target_average_feature_gradients)))
    left_operator = target_average_feature_gradients
    task_weights_gradients = []
    for x in range(len(args.auxiliary_task_list)):
        source_average_feature_gradients = source_average_feature_gradients_list[x]
        task_weights_gradients.append(-get_cosine(left_operator, source_average_feature_gradients))
    task_weights_gradients = np.array(task_weights_gradients)
    print('task weights gradients', task_weights_gradients)
    if args.normalized_gradients:
        # if np.min(task_weights_gradients) < 0:
            # task_weights_gradients /= np.abs(np.min(task_weights_gradients))
        # task_weights_gradients *= 1
        # max_interval = np.max(task_weights_gradients) - np.min(task_weights_gradients)
        # print('max interval', max_interval)
        # task_weights_gradients /= max_interval
        print('normalized task weights gradients', task_weights_gradients)
    else:
        '''
        if args.inv_constant == 0.0:
            if np.min(task_weights_gradients) < 0:
                # args.inv_constant = 1 / np.abs(np.min(task_weights_gradients))
                args.inv_constant = 1
            else:
                args.inv_constant = 1
        '''
        task_weights_gradients *= args.inv_constant
        print('inv constant', args.inv_constant)
        print('unnormalized task weights gradients', task_weights_gradients)
    return task_weights_gradients

