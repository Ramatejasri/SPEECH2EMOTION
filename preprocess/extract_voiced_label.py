import os
import sys
import pickle
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
from tqdm import tqdm
import librosa
sys.path.append('../')
import math
import random
import pandas as pd
import IPython.display
import librosa.display
import traceback
ms.use('seaborn-muted')
# %matplotlib inline
from models.utils import *

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

data_dir = '/home/b_ramatejasri258/SPEECH2EMOTION/data/'
path = '/home/b_ramatejasri258/SPEECH2EMOTION/data/pre-processed/'
labels_path = '{}df_iemocap.csv'.format(data_dir)
audio_vectors_path = '{}audio_vectors_mfcc_20_'.format(path)

labels_df = pd.read_csv(labels_path)

columns = ['wav_file', 'label', 'voiced']
# columns.extend(['feat_'+str(i) for i in range(20)])
df_features = pd.DataFrame(columns=columns)


for sess in (range(1, 2)):
        audio_vectors = pickle.load(open('{}{}.pkl'.format(audio_vectors_path, sess), 'rb'))
        for index, row in tqdm(labels_df[labels_df['wav_file'].str.contains('Ses0{}'.format(sess))].iterrows()):
            try:
                wav_file_name = row['wav_file']
                label = emotion_dict[row['emotion']]
                y = audio_vectors[wav_file_name]
                feat = [wav_file_name, label]
                # zcr = librosa.feature.zero_crossing_rate(y=y)
                # feat.extend(short_time_energy_and_zero_cross_rate(y))
                f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                print('voiced_flag', voiced_flag)
                feat.extend(voiced_flag)
                
                df_features = df_features.append(pd.DataFrame(feat, index=columns).transpose(), ignore_index=True)

            except:
                traceback.print_exc()
                print('Some exception occured')

df_features.to_csv('/home/b_ramatejasri258/SPEECH2EMOTION/data/pre-processed/audio_voiced_feats_librosa.csv', index=False)