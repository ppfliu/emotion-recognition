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


class GgfLayer(Layer):

    def __init__(self, output_dim, kernel_initializer='glorot_uniform',
                 kernel_regularizer=None, **kwargs):
        self.output_dim = output_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        super(GgfLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Wv = self.add_weight(name='Wv',
                                  shape=(int(input_shape[0][1]), self.output_dim),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  trainable=True)
        self.Wfv = self.add_weight(name='Wfv',
                                  shape=(int(input_shape[2][1]), self.output_dim),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  trainable=True)
        self.Wt = self.add_weight(name='Wt',
                                  shape=(int(input_shape[1][1]), self.output_dim),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  trainable=True)
        self.Wft = self.add_weight(name='Wft',
                                  shape=(int(input_shape[3][1]), self.output_dim),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  trainable=True)

        dim_size1 = int(input_shape[0][1]) + int(input_shape[1][1])
        dim_size2 = int(input_shape[2][1]) + int(input_shape[3][1])

        self.Wz_xv = self.add_weight(name='Wz_xv',
                                  shape=(dim_size1, self.output_dim),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  trainable=True)

        self.Wz_fv = self.add_weight(name='Wz_fv',
                                  shape=(dim_size2, self.output_dim),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  trainable=True)
        super(GgfLayer, self).build(input_shape)

    def call(self, inputs):
        x_v, x_t, f_v, f_t = inputs

        h_v = K.tanh(K.dot(x_v, self.Wv))
        h_t = K.tanh(K.dot(x_t, self.Wt))
        h_fv = K.tanh(K.dot(f_v, self.Wfv))
        h_ft = K.tanh(K.dot(f_t, self.Wft))

        z_xv = K.sigmoid(K.dot(K.concatenate([x_v, x_t]), self.Wz_xv))
        z_fv = K.sigmoid(K.dot(K.concatenate([f_v, f_t]), self.Wz_fv))

        if not K.learning_phase():
            z_xv = K.print_tensor(z_xv, message='z_xv:\n')
            z_fv = K.print_tensor(z_fv, message='z_fv:\n')

        h = z_xv * h_v + (1-z_xv) * h_t + z_fv * h_fv + (1-z_fv) * h_ft

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
        base_config = super(GgfLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Model

    input1 = Input(shape=(784,))
    input2 = Input(shape=(784,))
    input3 = Input(shape=(784,))
    input4 = Input(shape=(784,))

    feat1 = Dense(128, activation='relu')(input1)
    feat2 = Dense(64, activation='relu')(input2)
    feat3 = Dense(128, activation='relu')(input3)
    feat4 = Dense(64, activation='relu')(input4)

    gated_fusion = GgfLayer(output_dim=32)([feat1, feat2, feat3, feat4])

    predictions = Dense(10, activation='softmax')(gated_fusion)

    model = Model(inputs=[input1, input2, input3, input4], outputs=predictions)
    model.summary()
