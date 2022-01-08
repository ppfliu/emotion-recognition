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
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras.layers import Layer


class TflLayer(Layer):

    def __init__(self, **kwargs):
        super(TflLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(TflLayer, self).build(input_shape)

    def call(self, inputs):
        x_v = inputs[0]
        x_t = inputs[1]
        print(x_v.shape)
        print(x_t.shape)

        x_v_1 = tf.pad(x_v, paddings=[[0, 0], [0, 1]], mode='CONSTANT')
        x_t_1 = tf.pad(x_t, paddings=[[0, 0], [0, 1]], mode='CONSTANT')

        print(x_v_1.shape)
        print(x_t_1.shape)

        x_v_1_expand = tf.expand_dims(x_v_1, 2)
        x_t_1_expand = tf.expand_dims(x_t_1, 1)
        outer_prod = tf.matmul(x_v_1_expand, x_t_1_expand)
        print(outer_prod.shape)

        return outer_prod

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        print(input_shape)
        return (int(input_shape[0][0]), int(input_shape[0][1]), int(input_shape[0][1]))

    def get_config(self):
        base_config = super(TflLayer, self).get_config()
        return dict(list(base_config.items()))


if __name__ == '__main__':
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.models import Model

    input1 = Input(shape=(784,))
    input2 = Input(shape=(784,))

    feat1 = Dense(128, activation='relu')(input1)
    feat2 = Dense(64, activation='relu')(input2)

    tensor_fusion = TflLayer()([feat1, feat2])
    flatten = Flatten()(tensor_fusion)

    predictions = Dense(10, activation='softmax')(flatten)

    model = Model(inputs=[input1, input2], outputs=predictions)
    model.summary()
