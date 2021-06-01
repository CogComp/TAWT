import sklearn
from sklearn import metrics
from seqeval.metrics import f1_score, precision_score, recall_score
from scipy import stats


def get_label_data(file):
    data = []
    fin = open(file)
    lines = fin.readlines()
    label_seq = []
    for index in range(len(lines)):
        line = lines[index]
        line = line.strip()
        if len(line) == 0:
            if len(label_seq) != 0:
                data.append(label_seq)
                label_seq = []
        else:
            items = line.split()
            label_seq.append(items[1])
    print('total examples', len(data))
    return data


def get_performance(gold_data, pred_data, task):
    out_label_list = gold_data
    preds_list = pred_data
    if task == 'ner' or task == 'chunking':
        results = {
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
            "accuracy": sklearn.metrics.accuracy_score(y_label, y_pred),
            "micro f1": sklearn.metrics.f1_score(y_label, y_pred, average='micro'),
            "macro f1": sklearn.metrics.f1_score(y_label, y_pred, average='macro'),
        }
    return results


def get_instance_level_performance(gold_data, pred_data, task):
    score_list = []
    assert len(gold_data) == len(pred_data)
    for x in range(len(gold_data)):
        gold_label_seq = gold_data[x]
        pred_label_seq = pred_data[x]
        assert len(gold_label_seq) == len(pred_label_seq)
        if task == 'ner' or task == 'chunking':
            score = f1_score(gold_label_seq, pred_label_seq)
        elif task == 'predicate':
            score = sklearn.metrics.f1_score(gold_label_seq, pred_label_seq, pos_label='predicate')
        else:
            score = sklearn.metrics.accuracy_score(gold_label_seq, pred_label_seq)
        score_list.append(score)
    return score_list


def significance_testing(gold_file, pred_file, weighted_pred_file, task):
    gold_data = get_label_data(gold_file)
    pred_data = get_label_data(pred_file)
    weighted_pred_data = get_label_data(weighted_pred_file)
    results = get_performance(gold_data, pred_data, task)
    weighted_results = get_performance(gold_data, weighted_pred_data, task)
    for key in sorted(results.keys()):
        print("{0} = {1}".format(key, str(results[key])))
    for key in sorted(weighted_results.keys()):
        print("{0} = {1}".format(key, str(weighted_results[key])))
    score_list = get_instance_level_performance(gold_data, pred_data, task)
    weighted_score_list = get_instance_level_performance(gold_data, weighted_pred_data, task)
    ttest, pval = stats.ttest_rel(score_list, weighted_score_list)
    return pval


if __name__ == '__main__':
    dir_path = '/path/to/working/dir/'
    tasks = ['pos', 'chunking', 'predicate', 'ner']
    main_task_list = ['pos', 'chunking', 'predicate', 'ner']
    weighted_pre_training_inv_constant = {'pos': '30', 'chunking': '1', 'predicate': '1', 'ner': '100'}
    training_framework_list = ['joint-training', 'pre-training', 'normalized-joint-training']
    for main_task in main_task_list:
        auxiliary_task_list = []
        for task in tasks:
            if task != main_task:
                auxiliary_task_list.append(task)
        gold_file = dir_path + 'data/test_' + main_task + '.txt'
        for training_framework in training_framework_list:
            if training_framework == 'normalized-joint-training':
                pred_file = dir_path + 'bert-base-cased-joint-training-fixed-weights-multi-' \
                            + '-'.join(auxiliary_task_list) + '-' + main_task + '-normalized/' + main_task \
                            + '_test_predictions.txt'
                weighted_pred_file = dir_path + 'bert-base-cased-weighted-joint-training-multi-' \
                                     + '-'.join(auxiliary_task_list) + '-' + main_task + '-normalized/' + main_task \
                                     + '_test_predictions.txt'
            elif training_framework == 'pre-training':
                pred_file = dir_path + 'bert-base-cased-' + training_framework + '-multi-' \
                            + '-'.join(auxiliary_task_list) + '-' + main_task + '/' + main_task \
                            + '_test_predictions.txt'
                weighted_pred_file = dir_path + 'bert-base-cased-weighted-' + training_framework + '-multi-' \
                                     + '-'.join(auxiliary_task_list) + '-' + main_task + '-' \
                                     + weighted_pre_training_inv_constant[main_task] + '/' + main_task \
                                     + '_test_predictions.txt'
            else:
                pred_file = dir_path + 'bert-base-cased-' + training_framework + '-multi-' \
                            + '-'.join(auxiliary_task_list) + '-' + main_task + '/' + main_task \
                            + '_test_predictions.txt'
                weighted_pred_file = dir_path + 'bert-base-cased-weighted-' + training_framework + '-multi-' \
                                     + '-'.join(auxiliary_task_list) + '-' + main_task + '/' + main_task \
                                     + '_test_predictions.txt'
            pval = significance_testing(gold_file, pred_file, weighted_pred_file, main_task)
            print('Pvalue for {0} on {1}: {2}\n'.format(training_framework, main_task, str(pval)))


