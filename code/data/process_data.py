import random
import numpy as np


def split_ontonotes_data(file, pos_file, ner_file):
    fin = open(file)
    fout_pos = open(pos_file, 'w')
    fout_ner = open(ner_file, 'w')
    lines = fin.readlines()
    for line in lines:
        line = line.rstrip()
        if len(line) == 0:
            fout_pos.write('\n')
            fout_ner.write('\n')
            continue
        items = line.split()
        if items[1] == '*':
            continue
        fout_pos.write(items[0] + ' ' + items[1] + '\n')
        fout_ner.write(items[0] + ' ' + items[3] + '\n')

    fin.close()
    fout_pos.close()
    fout_ner.close()


def get_chunking_data(file, chunking_file):
    fin = open(file)
    fout_chunking = open(chunking_file, 'w')
    lines = fin.readlines()
    for line in lines:
        line = line.rstrip()
        if len(line) == 0:
            fout_chunking.write('\n')
            continue
        items = line.split()
        fout_chunking.write(items[0] + ' ' + items[2] + '\n')
    fin.close()
    fout_chunking.close()


def get_predicate_data(file, predicate_file):
    fin = open(file)
    fout_predicate = open(predicate_file, 'w')
    lines = fin.readlines()
    for line in lines:
        line = line.rstrip()
        if len(line) == 0:
            fout_predicate.write('\n')
            continue
        items = line.split()
        if items[2] != '-':
            items[2] = 'predicate'
        fout_predicate.write(items[0] + ' ' + items[2] + '\n')
    fin.close()
    fout_predicate.close()


def sample_ontonotes_data(file, sample_file, sample_size, task):
    data = []
    fin = open(file)
    fout = open(sample_file, 'w')
    lines = fin.readlines()
    word_seq = []
    pos_seq = []
    ner_seq = []
    for index in range(len(lines)):
        line = lines[index]
        line = line.strip()
        if len(line) == 0:
            if len(word_seq) != 0:
                data.append((word_seq, pos_seq, ner_seq))
                word_seq = []
                pos_seq = []
                ner_seq = []
        else:
            items = line.split()
            if items[1] == '*':
                continue
            word_seq.append(items[0])
            pos_seq.append(items[1])
            ner_seq.append(items[3])
    print('total examples', len(data))
    index_list = list(range(len(data)))
    random.shuffle(index_list)
    for index in index_list[:sample_size]:
        word_seq, pos_seq, ner_seq = data[index]
        for x in range(len(word_seq)):
            if task == 'pos':
                fout.write(word_seq[x] + " " + pos_seq[x] + '\n')
            else:
                fout.write(word_seq[x] + " " + ner_seq[x] + '\n')
        fout.write('\n')
    fin.close()
    fout.close()


def sample_chunking_data(file, sample_file, sample_size):
    data = []
    fin = open(file)
    fout = open(sample_file, 'w')
    lines = fin.readlines()
    word_seq = []
    chunking_seq = []
    for index in range(len(lines)):
        line = lines[index]
        line = line.strip()
        if len(line) == 0:
            if len(word_seq) != 0:
                data.append((word_seq, chunking_seq))
                word_seq = []
                chunking_seq = []
        else:
            items = line.split()
            word_seq.append(items[0])
            chunking_seq.append(items[2])
    print('total examples', len(data))
    index_list = list(range(len(data)))
    random.shuffle(index_list)
    for index in index_list[:sample_size]:
        word_seq, chunking_seq = data[index]
        for x in range(len(word_seq)):
            fout.write(word_seq[x] + " " + chunking_seq[x] + '\n')
        fout.write('\n')
    fin.close()
    fout.close()


def sample_predicate_data(file, sample_file, sample_size):
    data = []
    fin = open(file)
    fout = open(sample_file, 'w')
    lines = fin.readlines()
    word_seq = []
    predicate_seq = []
    for index in range(len(lines)):
        line = lines[index]
        line = line.strip()
        if len(line) == 0:
            if len(word_seq) != 0:
                data.append((word_seq, predicate_seq))
                word_seq = []
                predicate_seq = []
        else:
            items = line.split()
            if items[2] != '-':
                items[2] = 'predicate'
            word_seq.append(items[0])
            predicate_seq.append(items[2])
    print('total examples', len(data))
    index_list = list(range(len(data)))
    random.shuffle(index_list)
    for index in index_list[:sample_size]:
        word_seq, predicate_seq = data[index]
        for x in range(len(word_seq)):
            fout.write(word_seq[x] + " " + predicate_seq[x] + '\n')
        fout.write('\n')
    fin.close()
    fout.close()


def get_chunking_evaluation_data(gold_file, pred_file, combined_file):
    fin_gold = open(gold_file)
    fin_pred = open(pred_file)
    fout_combined = open(combined_file, 'w')
    gold_lines = fin_gold.readlines()
    pred_lines = fin_pred.readlines()
    assert len(gold_lines) == len(pred_lines)
    for index in range(len(gold_lines)):
        gold_line = gold_lines[index]
        pred_line = pred_lines[index]
        gold_line = gold_line.rstrip()
        pred_line = pred_line.rstrip()
        if len(gold_line) == 0:
            fout_combined.write('\n')
            continue
        gold_items = gold_line.split()
        pred_items = pred_line.split()
        fout_combined.write(gold_items[0] + ' ' + gold_items[1] + ' ' + gold_items[2] + ' ' + pred_items[1] + '\n')
    fin_gold.close()
    fin_pred.close()
    fout_combined.close()


def get_chunking_stats(file):
    data = []
    fin = open(file)
    lines = fin.readlines()
    word_seq = []
    chunking_seq = []
    word_num = 0
    for index in range(len(lines)):
        line = lines[index]
        line = line.strip()
        if len(line) == 0:
            if len(word_seq) != 0:
                data.append((word_seq, chunking_seq))
                word_seq = []
                chunking_seq = []
        else:
            items = line.split()
            word_seq.append(items[0])
            chunking_seq.append(items[1])
            word_num += 1
    print('total examples', len(data))
    print('average sentence length', word_num / len(data))


if __name__ == '__main__':
    dir_path = '/path/to/working/dir'
    train_file = dir_path + '/data/train.txt'
    dev_file = dir_path + '/data/dev.txt'
    test_file = dir_path + '/data/test.txt'
    train_pos_file = dir_path + '/data/train_pos.txt.tmp'
    dev_pos_file = dir_path + '/data/dev_pos.txt.tmp'
    test_pos_file = dir_path + '/data/test_pos.txt.tmp'
    train_ner_file = dir_path + '/data/train_ner.txt.tmp'
    dev_ner_file = dir_path + '/data/dev_ner.txt.tmp'
    test_ner_file = dir_path + '/data/test_ner.txt.tmp'
    sample_ontonotes_data(train_file, train_pos_file, 100000, 'pos')
    sample_ontonotes_data(train_file, train_ner_file, 100000, 'ner')
    # split_ontonotes_data(train_file, train_pos_file, train_ner_file)
    # split_ontonotes_data(dev_file, dev_pos_file, dev_ner_file)
    # split_ontonotes_data(test_file, test_pos_file, test_ner_file)

    orig_train_chunking_file = dir_path + '/data/conll2000/train.txt'
    orig_test_chunking_file = dir_path + '/data/conll2000/test.txt'
    train_chunking_file = dir_path + '/data/train_chunking.txt.tmp'
    test_chunking_file = dir_path + '/data/test_chunking.txt.tmp'
    test_chunking_pred_file = dir_path + '/bert-base-cased-single-chunking/test_predictions.txt'
    test_chunking_evaluation_file = dir_path + '/bert-base-cased-single-chunking/test_chunking_evaluation.txt'
    # sample_chunking_data(orig_train_chunking_file, train_chunking_file, 9000)
    # get_chunking_data(orig_test_chunking_file, test_chunking_file)

    # get_chunking_evaluation_data(orig_test_chunking_file, test_chunking_pred_file, test_chunking_evaluation_file)

    # get_chunking_stats(train_chunking_file)
    # get_chunking_stats(test_chunking_file)

    orig_train_predicate_file = dir_path + '/data/ontonotes_train.txt'
    orig_dev_predicate_file = dir_path + '/data/ontonotes_development.txt'
    orig_test_predicate_file = dir_path + '/data/ontonotes_test.txt'
    train_predicate_file = dir_path + '/data/train_predicate.txt.tmp'
    dev_predicate_file = dir_path + '/data/dev_predicate.txt.tmp'
    test_predicate_file = dir_path + '/data/test_predicate.txt.tmp'
    sample_predicate_data(orig_train_predicate_file, train_predicate_file, 100000)
    # get_predicate_data(orig_dev_predicate_file, dev_predicate_file)
    # get_predicate_data(orig_test_predicate_file, test_predicate_file)


