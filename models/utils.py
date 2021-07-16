import sys
import torch
import numpy as np
import pandas as pd
sys.path.append('../')
from models.config import model_config as config

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

import itertools
from collections import Counter
import matplotlib.pyplot as plt


def load_data(batched=True, test=False, file_dir='../preprocess/data/s2e/', model_type='full'):
    df = pd.read_csv('../preprocess/data/pre-processed/voiced_cls.csv')
    voiced = list(df.loc[df['voiced'] == True, 'wav_file'])
    unvoiced = list(df.loc[df['voiced'] == False, 'wav_file'])

    df_val = pd.read_csv('../preprocess/data/pre-processed/val_cls.csv')
    pos_val = list(df_val.loc[df_val['val'] == 'pos', 'wav_file'])
    neg_val = list(df_val.loc[df_val['val'] == 'neg', 'wav_file'])
    print(len(voiced),len(unvoiced), len(pos_val), len(neg_val))


    bs = config['batch_size']
    ftype = 'test' if test else 'train'
    # df = pd.read_csv('{}modified_df_{}.csv'.format(file_dir, ftype))
    df = pd.read_csv('{}audio_{}.csv'.format(file_dir, ftype))
    
    # 0th index in label, rest all are features
    data = (np.array(df[df.columns[:]]), np.array(df[df.columns[0]]))
    
    if test or not batched:
        new_data = []
        new_labels = []
        for d in data[0]:
            if model_type == 'full':
                new_data.append(d[2:])
                new_labels.append(d[1])
            elif model_type == 'val_based':
                if d[0] in unvoiced and d[0] in neg_val:
                    new_data.append(d[2:])
                    new_labels.append(d[1])
                if d[0] in voiced and d[0] in pos_val:
                    new_data.append(d[2:])
                    new_labels.append(d[1])
            elif model_type == 'voiced':
                if d[0] in voiced:
                    new_data.append(d[2:])
                    new_labels.append(d[1])
            elif model_type == 'unvoiced':
                if d[0] in unvoiced:
                    new_data.append(d[2:])
                    new_labels.append(d[1])
            elif model_type == 'neg_val_unvoiced':
                if d[0] in unvoiced and d[0] in neg_val:
                    new_data.append(d[2:])
                    new_labels.append(d[1])
            elif model_type == 'pos_val_voiced':
                if d[0] in voiced and d[0] in pos_val:
                    new_data.append(d[2:])
                    new_labels.append(d[1])
            new_data.append(d[2:])
            new_labels.append(d[1])
        print('test:',Counter(new_labels))
        # return [torch.FloatTensor(data[0]), torch.LongTensor(data[1])]
        return [torch.FloatTensor(new_data), torch.LongTensor(new_labels)]
    # exits()
    data = list(zip(data[0], data[1]))
    
    n_iters = len(data) // bs
    batches = []
    c_i,c_l = [], []

    for i in range(1, n_iters + 1):
        input_batch = []
        output_batch = []
        for e in data[bs * (i-1):bs * i]:
            # print(e)
            # input_batch.append(e[0][2:])
            # output_batch.append(e[0][1])
            # c_l.append(e[0][1])
            if e[0][0] in unvoiced and e[0][0] in neg_val:
                input_batch.append(e[0][2:])
                output_batch.append(e[0][1])
                c_l.append(e[0][1])
            if e[0][0] in voiced and e[0][0] in pos_val:
                input_batch.append(e[0][2:])
                output_batch.append(e[0][1])
                c_l.append(e[0][1])

        print(len(input_batch))
        print(len(output_batch))
        batches.append([torch.FloatTensor(input_batch),
                        torch.LongTensor(output_batch)])
    print('train:',Counter(c_l))

    return batches


def evaluate(targets, predictions):
    performance = {
        'acc': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average='macro'),
        'precision': precision_score(targets, predictions, average='macro'),
        'recall': recall_score(targets, predictions, average='macro')}
    return performance


def plot_confusion_matrix(targets, predictions, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # plt.figure(figsize=(8,8))
    cm = confusion_matrix(targets, predictions)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
