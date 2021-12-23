# %matplotlib inline

from typing import Dict, List, Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, Add
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Nadam

import loggers as lg
from abstractgame import AbstractGameState
from initialise import run_folder, run_archive_folder
from loss import softmax_cross_entropy_with_logits


# noinspection PyPep8Naming
class Gen_Model:
    def __init__(self, reg_const: float, learning_rate: float, input_dim: tf.Tensor, output_dim: int) -> None:
        self.reg_const: float = reg_const
        self.learning_rate: float = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model: Optional[Model] = None

    def predict(self, x: tf.Tensor) -> List[tf.Tensor]:
        return self.model.predict(x)

    def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
        return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split=validation_split,
                              batch_size=batch_size)

    def write(self, _, version):
        self.model.save(f"{run_folder}/models/version{version:0>4}.h5")

    @staticmethod
    def read(game, version):
        return load_model(
            f"{run_archive_folder}/{game}/models/version{version:0>4}.h5",
            custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})

    def print_weight_averages(self):
        layers = self.model.layers
        for i, l in enumerate(layers):
            try:
                x = l.get_weights()[0]
                lg.logger_model.info('WEIGHT LAYER %d: ABSAV = %f, SD =%f, ABSMAX =%f, ABSMIN =%f', i,
                                     np.mean(np.abs(x)), np.std(x), np.max(np.abs(x)), np.min(np.abs(x)))
            except Exception:
                pass
        lg.logger_model.info('------------------')
        for i, l in enumerate(layers):
            try:
                x = l.get_weights()[1]
                lg.logger_model.info('BIAS LAYER %d: ABSAV = %f, SD =%f, ABSMAX =%f, ABSMIN =%f', i, np.mean(np.abs(x)),
                                     np.std(x), np.max(np.abs(x)), np.min(np.abs(x)))
            except Exception:
                pass
        lg.logger_model.info('******************')

    def view_layers(self):
        layers = self.model.layers
        for i, l in enumerate(layers):
            x = l.get_weights()
            print('LAYER ' + str(i))

            try:
                weights = x[0]
                s = weights.shape

                fig = plt.figure(figsize=(s[2], s[3]))  # width, height in inches
                channel = 0
                filter_ = 0
                for j in range(s[2] * s[3]):
                    sub = fig.add_subplot(s[3], s[2], j + 1)
                    sub.imshow(weights[:, :, channel, filter_], cmap='coolwarm', clim=(-1, 1), aspect="auto")
                    channel = (channel + 1) % s[2]
                    filter_ = (filter_ + 1) % s[3]

            except Exception:
                try:
                    fig = plt.figure(figsize=(3, len(x)))  # width, height in inches
                    for j in range(len(x)):
                        sub = fig.add_subplot(len(x), 1, j + 1)
                        clim = (0, 2)
                        sub.imshow([x[i]], cmap='coolwarm', clim=clim, aspect="auto")

                    plt.show()

                except Exception:
                    try:
                        fig = plt.figure(figsize=(3, 3))  # width, height in inches
                        sub = fig.add_subplot(1, 1, 1)
                        sub.imshow(x[0], cmap='coolwarm', clim=(-1, 1), aspect="auto")

                        plt.show()

                    except Exception:
                        pass

            plt.show()

        lg.logger_model.info('------------------')


# noinspection PyPep8Naming
class Residual_CNN(Gen_Model):
    def __init__(self, reg_const: float, learning_rate: float, input_dim: tf.Tensor, output_dim: int,
                 hidden_layers: List[Dict[str, Union[int, Tuple[int, int]]]]) -> None:
        Gen_Model.__init__(self, reg_const, learning_rate, input_dim, output_dim)
        self.hidden_layers = hidden_layers
        self.num_layers: int = len(hidden_layers)
        self.model = self._build_model()

    def residual_layer(self, input_block: tensorflow.keras.Model, n_filters: int,
                       kernel_size: Tuple[int, int]) -> tensorflow.keras.Model:
        x = self.conv_layer(input_block, n_filters, kernel_size)
        x = Conv2D(
            filters=n_filters,
            kernel_size=kernel_size,
            data_format="channels_last",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([input_block, x])
        x = LeakyReLU()(x)

        return x

    def conv_layer(self, x: tensorflow.keras.Model, n_filters: int,
                   kernel_size: Tuple[int, int]) -> tensorflow.keras.Model:
        x = Conv2D(
            filters=n_filters,
            kernel_size=kernel_size,
            data_format="channels_last",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU()(x)

        return x

    def value_head(self, x: tensorflow.keras.Model) -> tensorflow.keras.Model:
        x = Conv2D(
            filters=1,
            kernel_size=(1, 1),
            data_format="channels_last",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)

        x = Dense(
            20,
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)

        x = LeakyReLU()(x)
        x = Dense(
            1,
            use_bias=False,
            activation='tanh',
            kernel_regularizer=regularizers.l2(self.reg_const),
            name='value_head'
        )(x)
        return x

    def policy_head(self, x: tensorflow.keras.Model) -> tensorflow.keras.Model:
        x = Conv2D(
            filters=2,
            kernel_size=(1, 1),
            data_format="channels_last",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const)
        )(x)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)

        x = Dense(
            self.output_dim,
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.reg_const),
            name='policy_head'
        )(x)

        return x

    def _build_model(self) -> Model:

        main_input = Input(shape=self.input_dim, name='main_input')

        x = tf.transpose(main_input, [0, 2, 3, 1])
        x = self.conv_layer(x, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])

        if len(self.hidden_layers) > 1:
            for h in self.hidden_layers[1:]:
                x = self.residual_layer(x, h['filters'], h['kernel_size'])

        vh = self.value_head(x)
        ph = self.policy_head(x)

        model = Model(inputs=[main_input], outputs=[vh, ph])
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
                      optimizer=Nadam(learning_rate=self.learning_rate),
                      loss_weights={'value_head': 0.2, 'policy_head': 0.8}
                      )

        return model

    def convert_to_model_input(self, state: AbstractGameState) -> tf.Tensor:
        input_to_model = state.binary
        input_to_model = tf.reshape(input_to_model, self.input_dim)
        return input_to_model
