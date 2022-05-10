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

import os
import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import AdditiveAttention as Attention
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Lambda

from tensorflow.keras import optimizers as opt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model


from gmu import GmuLayer
from ggf import GgfLayer
from tfl import TflLayer


GLOVE_DIR = './glove'
WORD_EMBED_DIM = 300

METRICS = [
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.Accuracy()
]


def prepare_embeddings(vocab_words, max_words):
    embeddings_index = {}
    #with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    with open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt')) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    num_words = min(max_words, len(vocab_words) + 1)
    embedding_matrix = np.zeros((num_words, WORD_EMBED_DIM))
    for word, i in vocab_words.items():
        if i >= max_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            if embedding_matrix[i].shape == embedding_vector.shape:
                embedding_matrix[i] = embedding_vector
            else:
                print(embedding_matrix[i].shape)
                print(embedding_vector.shape)
                embedding_matrix[i] = np.random.uniform(-0.25, 0.25)
        else:
            # Draw uniform vectors for words not found in embedding index
            embedding_matrix[i] = np.random.uniform(-0.25, 0.25)
            print(word, i)

    return embedding_matrix


def build_model(num_classes, vocab_words, max_words, max_sent_len, audio_input_shape, source, fusion):
    if 'audio' in source:
        ## Begin of 1D Conv
        audio_in = Input(shape=audio_input_shape)
        x_audio = audio_in
        for i in range(1, 5):
            filters, kernel_size = 4 + i * 8, 4
            x_audio = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(x_audio)
            x_audio = MaxPooling1D(pool_size=2)(x_audio)
        # End of 1D Conv

        x_audio_seq  = Bidirectional(LSTM(128, return_sequences=True))(x_audio)
        x_audio_vector = Lambda(lambda x: x[:, -1, :])(x_audio_seq)

    if 'text' in source:
        # text input
        text_in = Input(shape=(None,))
        x_text = Embedding(max_words, max_sent_len)(text_in)

        embedding_matrix = prepare_embeddings(vocab_words, max_words)
        x_text = Embedding(input_dim=max_words,
                           output_dim=WORD_EMBED_DIM,
                           embeddings_initializer=Constant(embedding_matrix),
                           input_length=max_sent_len, trainable=True)(text_in)

        x_convs = []
        for kernel in range(2, 5):
            x_conv = Conv1D(filters=128, kernel_size=kernel, padding='same', activation='relu', strides=1)(x_text)
            x_conv = MaxPooling1D(pool_size=2)(x_conv)
            x_convs.append(x_conv)

        x_text = concatenate(x_convs)

        x_text_seq = Bidirectional(LSTM(128, return_sequences=True))(x_text)
        x_text_vector = Lambda(lambda x: x[:, -1, :])(x_text_seq)

    if source == 'audio_text':
        if fusion == 'gmu':
            x_audio_text = GmuLayer(output_dim=128)([x_audio_vector, x_text_vector])
        elif fusion == 'tfl':
            x_audio_text = TflLayer()([x_audio_vector, x_text_vector])
            x_audio_text = Flatten()(x_audio_text)
        elif fusion == 'concat':
            x_audio_text = concatenate([x_audio_vector, x_text_vector], axis=-1)
        elif fusion == 'concat-attention':
            att_at = Attention()([x_audio_seq, x_text_seq])
            att_at_pool = GlobalAveragePooling1D()(att_at)

            att_ta = Attention()([x_text_seq, x_audio_seq])
            att_ta_pool = GlobalAveragePooling1D()(att_ta)

            x_audio_text = concatenate([att_at_pool, att_ta_pool], axis=-1)
        elif fusion == 'concat-all':
            att_at = Attention()([x_audio_seq, x_text_seq])
            att_at_pool = GlobalAveragePooling1D()(att_at)

            att_ta = Attention()([x_text_seq, x_audio_seq])
            att_ta_pool = GlobalAveragePooling1D()(att_ta)

            x_audio_text = concatenate([x_audio_vector, x_text_vector, att_at_pool, att_ta_pool], axis=-1)
        elif fusion == 'ggf':
            att_at = Attention()([x_audio_seq, x_text_seq])
            att_at_pool = GlobalAveragePooling1D()(att_at)
            #att_at_pool = Dropout(0.5)(att_at_pool)

            att_ta = Attention()([x_text_seq, x_audio_seq])
            att_ta_pool = GlobalAveragePooling1D()(att_ta)
            #att_ta_pool = Dropout(0.5)(att_ta_pool)

            x_audio_text = GgfLayer(output_dim=128, name='gmu_gate')([att_at_pool, att_ta_pool, x_text_vector, x_audio_vector])
        else:
            print('Unsupported fusion: {}'.format(fusion))
            sys.exit(-1)

        # apply dropout before Dense layer
        x_audio_text = Dense(128, activation='relu')(x_audio_text)
        x_audio_text = Dropout(0.5)(x_audio_text)

        output = Dense(num_classes, activation='softmax', name='output')(x_audio_text)
        model = Model([audio_in, text_in], output)
    elif source == 'audio':
        x_audio_vector = Dense(128, activation='relu')(x_audio_vector)
        x_audio_vector = Dropout(0.5)(x_audio_vector)
        output = Dense(num_classes, activation='softmax', name='output')(x_audio_vector)
        model = Model(audio_in, output)
    elif source == 'text':
        x_text_vector = Dense(128, activation='relu')(x_text_vector)
        x_text_vector = Dropout(0.5)(x_text_vector)
        output = Dense(num_classes, activation='softmax')(x_text_vector)
        model = Model(text_in, output)
    else:
        print('Unsupported source: {}'.format(source))
        sys.exit(-1)

    # adding regularization
    regularizer = l2(0.01)
    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    optimizer = opt.Adam(lr=0.0001, beta_1=0.9, beta_2=0.9, clipnorm=5.0)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', ])
    model.summary()

    return model
