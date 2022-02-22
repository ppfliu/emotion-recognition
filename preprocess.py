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

import os
import re
import sys
import pickle
import unicodedata
import pandas as pd
import config as cfg

from keras.preprocessing.text import Tokenizer


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def get_filepath(sess, filename):
    parent_dir = filename[:filename.rindex('_')]
    filepath = 'IEMOCAP_full_release/Session{}/sentences/wav/{}/{}.wav'.format(sess, parent_dir, filename)
    assert os.path.exists(filepath)
    return filepath


def extract_audio_labels(improvised):
    info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)

    start_times, end_times, wav_file_names, wav_paths, emotions, sessions, vals, acts, doms = \
        [], [], [], [], [], [], [], [], []

    for sess in range(1, 6):
        emo_evaluation_dir = 'IEMOCAP_full_release/Session{}/dialog/EmoEvaluation/'.format(sess)
        evaluation_files = [f for f in os.listdir(emo_evaluation_dir) if f.startswith('Ses')]
        for file in evaluation_files:
            with open(emo_evaluation_dir + file) as f:
                content = f.read()
            info_lines = re.findall(info_line, content)
            for line in info_lines[1:]:  # the first line is a header
                start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
                if improvised == 'true':
                    if not 'impro' in wav_file_name:
                        continue
                start_time, end_time = start_end_time[1:-1].split('-')
                val, act, dom = val_act_dom[1:-1].split(',')
                val, act, dom = float(val), float(act), float(dom)
                start_time, end_time = float(start_time), float(end_time)
                start_times.append(start_time)
                end_times.append(end_time)
                wav_file_names.append(wav_file_name)
                wav_paths.append(get_filepath(sess, wav_file_name))
                emotions.append(emotion)
                sessions.append(sess)
                vals.append(val)
                acts.append(act)
                doms.append(dom)

        df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'wav_path',
                                           'emotion', 'session', 'val', 'act', 'dom'])

    df_iemocap['start_time'] = start_times
    df_iemocap['end_time'] = end_times
    df_iemocap['wav_file'] = wav_file_names
    df_iemocap['wav_path'] = wav_paths
    df_iemocap['emotion'] = emotions
    df_iemocap['session'] = sessions
    df_iemocap['val'] = vals
    df_iemocap['act'] = acts
    df_iemocap['dom'] = doms

    df_iemocap.to_csv('df_iemocap.csv', index=False)

    return df_iemocap


def read_transcriptions():
    useful_regex = re.compile(r'^(\w+)', re.IGNORECASE)
    file2trans = {}

    for sess in range(1, 6):
        transcripts_path = 'IEMOCAP_full_release/Session{}/dialog/transcriptions/'.format(sess)
        transcript_files = [f for f in os.listdir(transcripts_path) if f.startswith('Ses')]
        for f in transcript_files:
            with open('{}{}'.format(transcripts_path, f), 'r') as f:
                all_lines = f.readlines()
            for line in all_lines:
                audio_code = useful_regex.match(line).group()
                transcription = line.split(':')[-1].strip()
                file2trans[audio_code] = transcription

    return file2trans


def dump_tokenzier(train_filename, tokenizer_path):
    df = pd.read_csv(train_filename)
    counter = {}
    max_sent_len = 0
    for idx, row in df.iterrows():
        words = row['text'].split()
        if len(words) > max_sent_len:
            max_sent_len = len(words)
        for word in words:
            if word in counter:
                counter[word] += 1
            else:
                counter[word] = 1

    top_words = []
    for word in counter:
        if counter[word] > 1:
            top_words.append(word)

    num_words = len(top_words) + 1
    tokenizer = Tokenizer(num_words=num_words, char_level=False, oov_token='_UNK_')
    tokenizer.fit_on_texts(df['text'].tolist())
    print('num_words:{}, len(word_index): {}'.format(num_words, len(tokenizer.word_index)))

    with open(tokenizer_path, 'wb') as tokenizer_file:
        pickle.dump({'tokenizer': tokenizer, 'maxlen': max_sent_len, 'num_words': num_words}, tokenizer_file, protocol=pickle.HIGHEST_PROTOCOL)


def main(test_sess, target_dir, improvised, split=None):
    df = extract_audio_labels(improvised)
    df_selected = df.loc[df['emotion'].isin(['hap', 'ang', 'sad', 'neu'])]

    file2trans = read_transcriptions()

    df_train = df_selected.loc[df['session'] != test_sess]
    save_dataset(file2trans, df_train, os.path.join(target_dir, 'train-{}.csv'.format(test_sess)))

    if split is None:
        df_test = df_selected.loc[df['session'] == test_sess]
        save_dataset(file2trans, df_test, os.path.join(target_dir, 'test-{}.csv'.format(test_sess)))
    else:
        df_valid = df_selected.loc[(df['session'] == test_sess) & (df['wav_file'].str.contains('_' + split[0]))]
        save_dataset(file2trans, df_valid, os.path.join(target_dir, 'valid-{}.csv'.format(test_sess)))

        df_test = df_selected.loc[(df['session'] == test_sess) & (df['wav_file'].str.contains('_' + split[1]))]
        save_dataset(file2trans, df_test, os.path.join(target_dir, 'test-{}.csv'.format(test_sess)))


def save_dataset(file2trans, df_data, filename):
    df_csv = pd.DataFrame()
    df_csv['session'] = df_data['session']
    df_csv['wav_file'] = df_data['wav_file']
    df_csv['wav_path'] = df_data['wav_path']
    df_csv['label'] = [cfg.EMOTION_CODE[emotion] for emotion in df_data['emotion']]
    df_csv['text'] = [normalizeString(file2trans[f]) for f in df_data['wav_file']]
    df_csv['start_time'] = df_data['start_time']
    df_csv['end_time'] = df_data['end_time']
    df_csv.to_csv(filename, index=False)
    print('dataset saved: {}'.format(filename))


if __name__ == '__main__':
    test_sess = int(sys.argv[1])
    targt_dir = sys.argv[2]
    improvised = sys.argv[3]
    if len(sys.argv) > 4:
        split = sys.argv[4]
    else:
        split = None
    main(test_sess, targt_dir, improvised, split)
    train_filename, tokenizer_path = targt_dir + '/train-{}.csv'.format(test_sess), 'tokenizer-{}.pkl'.format(test_sess)
    dump_tokenzier(train_filename, tokenizer_path)
