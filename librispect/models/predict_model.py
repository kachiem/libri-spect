import tensorflow as tf
import librosa
import deepspectrograminversion as dsi
from deepspectrograminversion.features import spectrogram
import glob
import numpy as np
from tensorflow import keras
from deepspectrograminversion.visualization import visualize

SPECT_HEIGHT = 2048
STFT_HEIGHT = 2048
WINDOW_SIZE = 16
STEP_SIZE = 4
BATCH_SIZE = 32
plot_loss = visualize.plotLoss()


class model_builder:

    def __init__(self, training_path_list, validation_path_list, window_size, spect_height,
                 stft_height, batch_size, step_size):
        self.window_size = window_size
        self.spect_height = spect_height
        self.stft_height = stft_height
        self.batch_size = batch_size
        self.step_size = step_size
        self.training_path_list = training_path_list
        self.validation_path_list = validation_path_list
        self.spect_maker = spectrogram.spect_maker(spectrogram.HPARAMS)
        self.inputs = keras.Input(
            shape=(SPECT_HEIGHT, WINDOW_SIZE), name='input')

    def train_model(self):
        self.model.fit_generator(generator=self.spect_maker.batch_iter(self.training_path_list, self.batch_size),
                                 steps_per_epoch=self.spect_maker.batch_ss_per_epoch(
                                     self.training_path_list, self.batch_size),
                                 validation_data=self.spect_maker.batch_iter(
                                     self.validation_path_list, self.batch_size),
                                 validation_steps=self.spect_maker.batch_ss_per_epoch(
                                     self.validation_path_list, self.batch_size),
                                 callbacks=[plot_loss],
                                 epochs=5)


class conv2d_model(model_builder):

    def __init__(self, training_path_list, validation_path_list, window_size, spect_height, stft_height, batch_size, step_size):
        super(conv2d_model, self).__init__(training_path_list, validation_path_list,
                                           window_size, spect_height, stft_height, batch_size, step_size)
        self.reshape = keras.layers.Reshape([SPECT_HEIGHT, WINDOW_SIZE, 1])
        self.conv1 = keras.layers.Conv2D(filters=16, kernel_size=[
                                         16, 16], activation='relu')
        self.flat = keras.layers.Flatten()
        self.layers = [
            keras.layers.Dense(128, activation='relu', name='dense1'),
            keras.layers.Dense(128, activation='relu', name='dense2'),
            keras.layers.Dense(STFT_HEIGHT * WINDOW_SIZE * 2,
                               activation='relu', name='output')
        ]
        self.outputs = self.get_model_output(self.reshape(self.inputs))
        self.stft_hat = keras.layers.Reshape(
            [2, STFT_HEIGHT, WINDOW_SIZE])(self.outputs[-1])
        self.model = keras.models.Model(inputs=self.inputs,
                                        outputs=self.stft_hat)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    def get_model_output(self, x):
        self.outputs = []
        x = self.conv1(x)
        x = self.flat(x)
        for self.layer in self.layers:
            x = self.layer(x)
            self.outputs.append(x)
        return self.outputs


class dense_model(model_builder):

    def __init__(self, training_path_list, validation_path_list, window_size, spect_height, stft_height, batch_size, step_size):
        super(dense_model, self).__init__(training_path_list, validation_path_list, window_size,
                                          spect_height, stft_height, batch_size, step_size)
        self.reshape = keras.layers.Reshape([SPECT_HEIGHT * WINDOW_SIZE])
        self.layers = [
            keras.layers.Dense(512, activation='relu', name='dense1'),
            keras.layers.Dense(512, activation='relu', name='dense2'),
            keras.layers.Dense(STFT_HEIGHT * WINDOW_SIZE * 2,
                               activation='relu', name='output')
        ]
        self.outputs = self.get_model_output(self.reshape(self.inputs))
        self.stft_hat = keras.layers.Reshape(
            [2, STFT_HEIGHT, WINDOW_SIZE])(self.outputs[-1])
        self.model = keras.models.Model(inputs=self.inputs,
                                        outputs=self.stft_hat)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    def get_model_output(self, x):
        self.outputs = []
        for self.layer in self.layers:
            x = self.layer(x)
            self.outputs.append(x)
        return self.outputs
