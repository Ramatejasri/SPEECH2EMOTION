import os
import sys
import pickle
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
from tqdm import tqdm
import librosa
import math
import random
import pandas as pd
import IPython.display
import librosa.display
import traceback
ms.use('seaborn-muted')
# %matplotlib inline

emotion_dict = {'ang': 0,
                'hap': 1,
                'exc': 2,
                'sad': 3,
                'fru': 4,
                'fea': 5,
                'sur': 6,
                'neu': 7,
                'xxx': 8,
                'oth': 8}

data_dir = 'data/pre-processed/'
labels_path = '{}df_iemocap.csv'.format(data_dir)
audio_vectors_path = '{}audio_vectors_'.format(data_dir)

labels_df = pd.read_csv(labels_path)

columns = ['wav_file', 'label']
columns.extend(['feat_'+str(i) for i in range(10)])
df_features = pd.DataFrame(columns=columns)

for sess in (range(1, 2)):
        audio_vectors = pickle.load(open('{}{}.pkl'.format(audio_vectors_path, sess), 'rb'))
        for index, row in tqdm(labels_df[labels_df['wav_file'].str.contains('Ses0{}'.format(sess))].iterrows()):
            try:
                wav_file_name = row['wav_file']
                label = emotion_dict[row['emotion']]
                y = audio_vectors[wav_file_name]
                feat = [wav_file_name, label]
                feat.extend(y)
                df_features = df_features.append(pd.DataFrame(feat, index=columns).transpose(), ignore_index=True)

            except:
                traceback.print_exc()
                print('Some exception occured')

df_features.to_csv('data/pre-processed/audio_features.csv', index=False)