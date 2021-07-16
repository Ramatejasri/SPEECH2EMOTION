#!/usr/bin/env python
import os

import scipy.io.wavfile as wav
from tqdm import tqdm
import librosa
import pandas as pd
import math
import pickle


labels_df = pd.read_csv('data/pre-processed/df_iemocap.csv')
iemocap_dir = '/home/b_ramatejasri258/project/data/IEMOCAP_full_release/'

sr = 44100
audio_vectors = {}
for sess in range(1,2):
    wav_file_path = '{}Session{}/dialog/wav/'.format(iemocap_dir, sess)
    orig_wav_files = os.listdir(wav_file_path)
    for orig_wav_file in tqdm(orig_wav_files):
        try:
            orig_wav_vector, _sr = librosa.load(wav_file_path + orig_wav_file, sr=sr)
            orig_wav_file, file_format = orig_wav_file.split('.')
            for index, row in labels_df[labels_df['wav_file'].str.contains(orig_wav_file)].iterrows():
                start_time, end_time, truncated_wav_file_name, emotion, val, act, dom = row['start_time'], row['end_time'], row['wav_file'], row['emotion'], row['val'], row['act'], row['dom']
                start_frame = math.floor(start_time * sr)
                end_frame = math.floor(end_time * sr)
                truncated_wav_vector = orig_wav_vector[start_frame:end_frame + 1]
                mfcc = librosa.feature.mfcc(y=truncated_wav_vector, sr=sr,n_mfcc=20)
                mfcc_fil = mfcc.mean(axis=1)
                audio_vectors[truncated_wav_file_name] = mfcc_fil

            
        except:
            print('An exception occured for {}'.format(orig_wav_file))
    with open('data/pre-processed/audio_vectors_{}.pkl'.format(sess), 'wb') as f:
        pickle.dump(audio_vectors, f)