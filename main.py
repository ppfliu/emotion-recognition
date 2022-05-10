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

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import pickle
import numpy as np
import random as rn

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Attention
from tensorflow.keras.layers import AdditiveAttention
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

from model import build_model
from data import load_data
from config import LABEL_NAMES

from gmu import GmuLayer
from tfl import TflLayer
from ggf import GgfLayer

## Begin of settings for reproducible results
import os
os.environ['PYTHONHASHSEED'] = '33'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(33)
rn.seed(33)
tf.set_random_seed(33)
#tf.random.set_seed(33)

sess_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                           inter_op_parallelism_threads=1)
sess_conf.gpu_options.allow_growth = True
sess_conf.log_device_placement = False
sess = tf.Session(graph=tf.get_default_graph(), config=sess_conf)
K.set_session(sess)
## End of settings

flags = tf.app.flags
flags.DEFINE_string(name='mode', default='train', help='set running mode: train, test')
flags.DEFINE_string(name='fusion', default='gmu', help='fusion model: concat, gmu, tfl')
flags.DEFINE_string(name='source', default='audio_text', help='data source: audio, text, audio_text')
flags.DEFINE_string(name='tokenizer_filename', default='tokenizer.pkl', help='file name of tokenizer')
flags.DEFINE_string(name='train_filename', default='./time-mfcc/train.npz', help='train filename of the features data')
flags.DEFINE_string(name='valid_filename', default=None, help='valid filename of the features data')
flags.DEFINE_string(name='test_filename', default='./time-mfcc/test.npz', help='test filename of the features data')
flags.DEFINE_integer(name='batch_size', default=32, help='number of examples in a batch')
flags.DEFINE_integer(name='epochs', default=100, help='number of epochs')
flags.DEFINE_integer(name='patience', default=5, help='patience before stopping training')


def train():
    source = flags.FLAGS.source
    fusion = flags.FLAGS.fusion
    train_filename = flags.FLAGS.train_filename
    valid_filename = flags.FLAGS.valid_filename
    batch_size = flags.FLAGS.batch_size
    epochs = flags.FLAGS.epochs
    patience = flags.FLAGS.patience

    x_train_audio, x_train_text, y_train = load_data(train_filename)
    #x_train_audio, x_train_text, y_train = shuffle(x_train_audio, x_train_text, y_train, random_state=28)

    if valid_filename is None:
        x_train_audio, x_valid_audio, x_train_text, x_valid_text, y_train, y_valid = train_test_split(
            x_train_audio, x_train_text, y_train, random_state=15, stratify=y_train, test_size=0.05)
    else:
        x_valid_audio, x_valid_text, y_valid = load_data(valid_filename)
    print('train:', x_train_audio.shape, x_train_text.shape, y_train.shape)
    print('valid:', x_valid_audio.shape, x_valid_text.shape, y_valid.shape)

    # compute class weights
    classes = np.unique(y_train)
    print(classes)
    weights = class_weight.compute_class_weight('balanced', classes, y_train)
    print(weights)

    # load tokenizer
    with open(flags.FLAGS.tokenizer_filename, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    model = build_model(len(classes), tokenizer['tokenizer'].word_index, tokenizer['num_words'], tokenizer['maxlen'],
                        audio_input_shape=(x_train_audio.shape[1], x_train_audio.shape[2]), source=source, fusion=fusion)

    early_stopping = EarlyStopping(monitor='val_acc', patience=patience)
    checkpoint_callback = ModelCheckpoint('models/{}-{}-best.h5'.format(source, fusion),
                                          verbose=1, save_best_only=True, monitor='val_acc', mode='max')

    if source == 'audio':
        model.fit(x_train_audio, y_train, batch_size=batch_size, epochs=epochs, class_weight=weights,
                  validation_data=(x_valid_audio, y_valid), verbose=1,
                  callbacks=[early_stopping, checkpoint_callback])
    elif source == 'text':
        model.fit(x_train_text, y_train, batch_size=batch_size, epochs=epochs, class_weight=weights,
                  validation_data=(x_valid_text, y_valid), verbose=1,
                  callbacks=[early_stopping, checkpoint_callback])
    else:
        model.fit([x_train_audio, x_train_text], y_train, batch_size=batch_size, epochs=epochs, class_weight=weights,
                  validation_data=([x_valid_audio, x_valid_text], y_valid), verbose=1,
                  callbacks=[early_stopping, checkpoint_callback])


def get_custom_objects():
    custom_objects = {'GmuLayer': GmuLayer, 'TflLayer': TflLayer, 'Attention': Attention,
                      'GgfLayer': GgfLayer, 'AdditiveAttention': AdditiveAttention}
    return custom_objects


def test():
    x_test_audio, x_test_text, y_test= load_data(flags.FLAGS.test_filename)
    print(x_test_audio.shape, x_test_text.shape, y_test.shape)

    source = flags.FLAGS.source
    fusion = flags.FLAGS.fusion
    model = load_model('models/{}-{}-best.h5'.format(source, fusion), custom_objects=get_custom_objects())

    if source == 'audio':
        scores = model.evaluate(x_test_audio, y_test)
        print('scores: ', scores)
        predictions = model.predict(x_test_audio).argmax(axis=-1)
    elif source == 'text':
        scores = model.evaluate(x_test_text, y_test)
        print('scores: ', scores)
        predictions = model.predict(x_test_text).argmax(axis=-1)
    else:
        scores = model.evaluate([x_test_audio, x_test_text], y_test)
        print('scores: ', scores)
        predictions = model.predict([x_test_audio, x_test_text]).argmax(axis=-1)

    print('accuracy: {:.4f}'.format(accuracy_score(y_test, predictions)))
    print(classification_report(y_test, predictions, digits=4, target_names=LABEL_NAMES))
    print(confusion_matrix(y_test, predictions))


if __name__ == '__main__':
    mode = flags.FLAGS.mode
    if mode == 'train':
        train()
    elif mode == 'test':
        test()
    else:
        print('Unsupported mode: {}'.format(mode))
