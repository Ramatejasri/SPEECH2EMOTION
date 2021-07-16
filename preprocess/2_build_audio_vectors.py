#!/usr/bin/env python
import os
from MFCC import mfcc
from MFCC import delta
from MFCC import logfbank
import scipy.io.wavfile as wav
from tqdm import tqdm
import librosa
import pandas as pd
import math
import pickle

emotion_dict = {'ang': 0,
                'hap': 1,
                'sad': 2,
                'neu': 3}

labels_df = pd.read_csv('/home/b_ramatejasri258/multimodal-speech-emotion-recognition/data/pre-processed/df_iemocap.csv')
# l=file_emo.loc[file_emo['wav_file'] == 'Ses01F_script02_1', 'emotion'].iloc[0]
# print(l)
# exit()
iemocap_dir = '/home/b_ramatejasri258/project/data/IEMOCAP_full_release/'
# (rate,sig) = wav.read("/home/b_ramatejasri258/project/data/IEMOCAP_full_release/Session1/dialog/wav/Ses01F_impro01.wav")
# y, _sr = librosa.load('/home/b_ramatejasri258/project/data/IEMOCAP_full_release/Session1/dialog/wav/Ses01F_impro01.wav')
# mfcc = librosa.feature.mfcc(y=y, sr=44100)
# mfcc_fil = mfcc.mean(axis=1)
# print(mfcc.shape)
# print(mfcc_fil.shape,mfcc_fil)
sr = 44100
audio_vectors = {}
for sess in range(5,6):  # using one session due to memory constraint, can replace [5] with range(1, 6)
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
                mfcc = librosa.feature.mfcc(y=truncated_wav_vector, sr=sr,n_mfcc=10)
                mfcc_fil = mfcc.mean(axis=1)
                audio_vectors[truncated_wav_file_name] = mfcc_fil

            
        except:
            print('An exception occured for {}'.format(orig_wav_file))
    with open('data/pre-processed/audio_vectors_{}.pkl'.format(sess), 'wb') as f:
        pickle.dump(audio_vectors, f)