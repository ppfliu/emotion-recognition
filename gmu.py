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

from __future__ import absolute_import, division, print_function


from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer


class GmuLayer(Layer):

    def __init__(self, output_dim, kernel_initializer='glorot_uniform',
                 kernel_regularizer=None, **kwargs):
        self.output_dim = output_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        super(GmuLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Wv = self.add_weight(name='Wv',
                                  shape=(int(input_shape[0][1]), self.output_dim),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  trainable=True)
        self.Wt = self.add_weight(name='Wt',
                                  shape=(int(input_shape[1][1]), self.output_dim),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  trainable=True)
        self.Wz = self.add_weight(name='Wz',
                                  shape=(int(input_shape[0][1]) + int(input_shape[1][1]), self.output_dim),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  trainable=True)
        super(GmuLayer, self).build(input_shape)

    def call(self, inputs):
        x_v = inputs[0]
        x_t = inputs[1]

        h_v = K.tanh(K.dot(x_v, self.Wv))
        h_t = K.tanh(K.dot(x_t, self.Wt))
        z = K.sigmoid(K.dot(K.concatenate([x_v, x_t]), self.Wz))
        h = z * h_v + (1-z) * h_t

        return h

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return (int(input_shape[0][0]), self.output_dim)


    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer)
        }
        base_config = super(GmuLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Model

    input1 = Input(shape=(784,))
    input2 = Input(shape=(784,))

    feat1 = Dense(128, activation='relu')(input1)
    feat2 = Dense(64, activation='relu')(input2)

    gated_fusion = GmuLayer(output_dim=32)([feat1, feat2])

    predictions = Dense(10, activation='softmax')(gated_fusion)

    model = Model(inputs=[input1, input2], outputs=predictions)
    model.summary()
