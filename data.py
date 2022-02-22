# Copyright (C) 2020 Pengfei Liu. All rights reserved.
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

import sys
import librosa
import pickle
import math
import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from scipy import signal
from imblearn.over_sampling import RandomOverSampler

from config import FEAT_MAX_LEN, MFCC_NUM, SAMPLING_RATE

# frame length: 25ms
frame_len = 0.025
window = 'hamming'
win_length = int(frame_len * SAMPLING_RATE)
# frame shift: 10ms
frame_shift = 0.01
hop_length = int(frame_shift * SAMPLING_RATE)
# DFT length
n_fft = 512
# number of mels
n_mels = 26 #128
# frequency range
fmin = 0
fmax = 6500 #5120


def log_spectrogram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))

    freqs, _, spec = signal.spectrogram(audio,
                                        nfft=3198,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)

    feats = np.swapaxes(np.log(spec.T.astype(np.float32) + eps)[:, :512], 1, 0)
    return pad_feats(feats)


def extract_spectrogram(yt, max_len=FEAT_MAX_LEN):
    duration = librosa.get_duration(y=yt, sr=SAMPLING_RATE)
    print('duration: ', duration)
    print('#frames: ', 1 + math.floor((len(yt)-400) / 160))

    mel_spec = librosa.feature.melspectrogram(yt, sr=SAMPLING_RATE, n_fft=n_fft,
                                           hop_length=hop_length, win_length=win_length,
                                           window=window, n_mels=n_mels, fmin=fmin, fmax=fmax)

    mel_spec = np.log(mel_spec + 1e-10)
    return pad_feats(mel_spec)


def extract_mfcc(yt, max_len=FEAT_MAX_LEN):
    duration = librosa.get_duration(y=yt, sr=SAMPLING_RATE)
    print('duration: ', duration)
    print('#frames: ', 1 + math.floor((len(yt)-400) / 160))

    mfcc = librosa.feature.mfcc(yt, sr=SAMPLING_RATE, n_mfcc=MFCC_NUM, n_fft=n_fft,
                                           hop_length=hop_length, win_length=win_length,
                                           window=window, n_mels=n_mels, fmin=fmin, fmax=fmax)

    print('mfcc: ', mfcc.shape)
    mfcc_delta = librosa.feature.delta(mfcc)
    print('mfcc_delta: ', mfcc_delta.shape)

    feats = np.concatenate((mfcc, mfcc_delta), axis=0)
    return pad_feats(feats)


def pad_feats(feats):
    print('orig feats: ', feats.shape)
    # If maximum length exceeds feats lengths then pad the remaining ones
    if FEAT_MAX_LEN > feats.shape[1]:
        pad_width = FEAT_MAX_LEN - feats.shape[1]
        feats = np.pad(feats, pad_width=((0, 0), (0, pad_width)), mode='constant')
    # Else cutoff the remaining parts
    else:
        feats = feats[:, :FEAT_MAX_LEN]

    feats = np.swapaxes(feats, 0, 1)
    print('padded feats: ', feats.shape)

    return feats


def prepare_sequence_data(csv_filename, tokenizer_filename, feat_name='spec', oversampling=False):
    feats = []
    texts = []
    labels = []

    df_csv = pd.read_csv(csv_filename)

    for index, row in df_csv.iterrows():
        print(index, row['wav_file'])
        yt, _ = librosa.load(row['wav_path'], sr=SAMPLING_RATE)
        if feat_name == 'mfcc':
            feat = extract_mfcc(yt)
        elif feat_name == 'spec':
            feat = extract_spectrogram(yt)
        else:
            spec = extract_spectrogram(yt)
            mfcc = extract_mfcc(yt)
            feat = np.concatenate((spec, mfcc), axis=1)

        feats.append(feat)
        texts.append(row['text'])
        labels.append(row['label'])

    with open(tokenizer_filename, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    text_data = tokenizer['tokenizer'].texts_to_sequences(texts)
    text_data = sequence.pad_sequences(text_data, maxlen=tokenizer['maxlen'])
    print('text_data shape:', text_data.shape)

    audio_data = np.asarray(feats)
    print('audio_data shape:', audio_data.shape)

    label_array = np.array(labels)
    print('label shape:', label_array.shape)

    print(audio_data.shape, label_array.shape)
    if oversampling:
        audio_data_reshape = np.reshape(audio_data, (len(audio_data), -1))
        text_data_reshape = np.reshape(text_data, (len(text_data), -1))
        audio_text = np.concatenate((audio_data_reshape, text_data_reshape), axis=-1)
        oversample = RandomOverSampler(sampling_strategy='not majority', random_state=20)
        audio_text_samples, label_samples = oversample.fit_resample(audio_text, label_array)

        audio_data_part = audio_text_samples[:, :audio_data_reshape.shape[1]]
        audio_data_part = np.reshape(audio_data_part, (len(audio_data_part), audio_data.shape[1], -1))
        text_data_part = audio_text_samples[:, audio_data_reshape.shape[1]:]
        text_data_part = np.reshape(text_data_part, (len(text_data_part), text_data.shape[1]))
        audio_data = audio_data_part
        text_data = text_data_part
        label_array = label_samples
        print('after oversampling:', audio_data.shape, text_data.shape, label_array.shape)

    return audio_data, text_data, label_array

def normalize(data):
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    data = (data-mean)/std
    return data

def load_data(file_name):
    data = np.load(file_name)
    return data['audio'], data['text'], data['labels']


if __name__ == '__main__':
    csv_filename, tokenizer_filename, target_file, feat_name = \
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    oversampling = False
    if len(sys.argv) > 5:
        if sys.argv[5] == 'true':
            oversampling = True

    print('oversampling:', oversampling)
    x_audio, x_text, y_labels = prepare_sequence_data(csv_filename, tokenizer_filename, feat_name, oversampling)

    np.savez(target_file, audio=x_audio, text=x_text, labels=y_labels)
