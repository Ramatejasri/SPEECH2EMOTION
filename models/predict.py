import torch
import sys
import pickle
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils import load_data, evaluate, plot_confusion_matrix
import pandas as pd
from config import model_config as config
from sklearn.metrics import classification_report
from utils import plot_confusion_matrix
from collections import Counter

from lstm_classifier import LSTMClassifier
test_df = pd.read_csv('/home/b_ramatejasri258/project/MFCC/data/s2e/audio_test.csv')
test = test_df.iloc[0]
sample = torch.FloatTensor(list(test)[2:])


# df = pd.read_csv('/home/b_ramatejasri258/project/voiced/data/pre-processed/audio_voiced_feats_librosa.csv')
# voiced = list(df.loc[df['voiced'] == True, 'wav_file'])
# unvoiced = list(df.loc[df['voiced'] == False, 'wav_file'])

df = pd.read_csv('/home/b_ramatejasri258/multimodal-speech-emotion-recognition/data/pre-processed/voice_unvoiced_based_on_val.csv')
voiced = list(df.loc[df['label'] == True, 'wav_file'])
unvoiced = list(df.loc[df['label'] == False, 'wav_file'])

df_val = pd.read_csv('/home/b_ramatejasri258/multimodal-speech-emotion-recognition/data/pre-processed/voice_unvoiced_based_on_val.csv')
pos_val = list(df_val.loc[df_val['label'] == True, 'wav_file'])
neg_val = list(df_val.loc[df_val['label'] == False, 'wav_file'])
print(len(voiced),len(unvoiced), len(pos_val), len(neg_val))

    # 0th index in label, rest all are features
data = (np.array(test_df[test_df.columns[:]]), np.array(test_df[test_df.columns[0]]))
print(len(data[0]))
new_data = []
new_labels = []
for d in data[0]:
    new_data.append(d[2:])
    new_labels.append(d[1])
    # if d[0] in voiced and d[0] in pos_val:
    #     new_data.append(d[2:])
    #     new_labels.append(d[1])
    # if d[0] in unvoiced and d[0] in neg_val:
    #     new_data.append(d[2:])
    #     new_labels.append(d[1])
# return [torch.FloatTensor(data[0]), torch.LongTensor(data[1])]
inputs,targets = [torch.FloatTensor(new_data), torch.LongTensor(new_labels)]
emotion_dict = {'ang': 0, 'hap': 1, 'sad': 2, 'fea': 3, 'sur': 4, 'neu': 5}
print(len(new_data))
print(Counter(new_labels))
device = 'cuda:{}'.format(config['gpu']) if \
            torch.cuda.is_available() else 'cpu'

model = LSTMClassifier(config)
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

model = model.to(device)
checkpoint = torch.load('/home/b_ramatejasri258/multimodal-speech-emotion-recognition/lstm_classifier/s2e/runs/basic_lstm-best_model_with_full.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()
out = model(inputs.unsqueeze(0))
y_pred = [torch.argmax(i).numpy() for i in out]
y_pred = np.array(y_pred)
plot_confusion_matrix(targets, y_pred, classes=['ang', 'hap', 'sad', 'fea', 'sur', 'neu'])
print('\nClassification Report\n')
print(classification_report(targets, y_pred, target_names=['ang', 'hap', 'sad', 'fea', 'sur', 'neu']))
