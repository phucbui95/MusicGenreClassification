# %load compact_cnn/models.py
# 2016-06-06 Updating for Keras 1.0 API
import numpy as np
import keras
import pandas as pd
import librosa
import os

from keras import backend as K
from argparse import Namespace
from keras.models import Sequential, Model
from keras.layers import Layer, Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.pooling import AveragePooling2D
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D

import matplotlib.pyplot as plt
import argparse

SR = 12000


def build_convnet_model(args, last_layer=True, sr=None, compile=True):
    ''' '''
    tf = args.tf_type
    normalize = args.normalize
    if normalize in ('no', 'False'):
        normalize = None
    decibel = args.decibel
    model = raw_vgg(args, tf=tf, normalize=normalize, decibel=decibel,
                    last_layer=last_layer, sr=sr)
    if compile:
        model.compile(optimizer=keras.optimizers.Adam(lr=5e-3),
                      loss='binary_crossentropy')
    return model


def raw_vgg(args, input_length=12000 * 29, tf='melgram', normalize=None,
            decibel=False, last_layer=True, sr=None):
    ''' when length = 12000*29 and 512/256 dft/hop,
    melgram size: (n_mels, 1360)
    '''
    assert tf in ('stft', 'melgram')
    assert normalize in (None, False, 'no', 0, 0.0, 'batch', 'data_sample', 'time', 'freq', 'channel')
    assert isinstance(decibel, bool)

    if sr is None:
        sr = SR  # assumes 12000

    conv_until = args.conv_until
    trainable_kernel = args.trainable_kernel
    model = Sequential()
    # decode args
    fmin = args.fmin
    fmax = args.fmax
    if fmax == 0.0:
        fmax = sr / 2
    n_mels = args.n_mels
    trainable_fb = args.trainable_fb
    model.add(Melspectrogram(n_dft=512, n_hop=256, power_melgram=2.0,
                             input_shape=(1, input_length),
                             trainable_kernel=trainable_kernel,
                             trainable_fb=trainable_fb,
                             return_decibel_melgram=decibel,
                             sr=sr, n_mels=n_mels,
                             fmin=fmin, fmax=fmax,
                             name='melgram'))

    poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (4, 4)]

    if normalize in ('batch', 'data_sample', 'time', 'freq', 'channel'):
        model.add(Normalization2D(normalize))
    model.add(get_convBNeluMPdrop(5, [32, 32, 32, 32, 32],
                                  [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
                                  poolings, model.output_shape[1:], conv_until=conv_until))
    if conv_until != 4:
        model.add(GlobalAveragePooling2D())
    else:
        model.add(Flatten())

    if last_layer:
        model.add(Dense(50, activation='sigmoid'))
    return model


def get_convBNeluMPdrop(num_conv_layers, nums_feat_maps,
                        conv_sizes, pool_sizes, input_shape, conv_until=None):
    # [Convolutional Layers]
    model = Sequential(name='ConvBNEluDr')
    input_shape_specified = False

    if conv_until is None:
        conv_until = num_conv_layers  # end-inclusive.

    for conv_idx in xrange(num_conv_layers):
        # add conv layer
        if not input_shape_specified:
            model.add(Convolution2D(nums_feat_maps[conv_idx],
                                    conv_sizes[conv_idx][0], conv_sizes[conv_idx][1],
                                    input_shape=input_shape,
                                    border_mode='same',
                                    init='he_normal'))
            input_shape_specified = True
        else:
            model.add(Convolution2D(nums_feat_maps[conv_idx],
                                    conv_sizes[conv_idx][0], conv_sizes[conv_idx][1],
                                    border_mode='same',
                                    init='he_normal'))
        # add BN, Activation, pooling
        model.add(BatchNormalization(axis=1, mode=2))
        model.add(keras.layers.advanced_activations.ELU(alpha=1.0))  # TODO: select activation

        model.add(MaxPooling2D(pool_size=pool_sizes[conv_idx]))
        if conv_idx == conv_until:
            break

    return model


def load_model(mode, conv_until=None):
    # setup stuff to build model

    # This is it. use melgram, up to 6000 (SR is assumed to be 12000, see model.py),
    # do decibel scaling
    assert mode in ('feature', 'tagger')
    if mode == 'feature':
        last_layer = False
    else:
        last_layer = True

    if conv_until is None:
        conv_until = 4

    assert K.image_dim_ordering() == 'th', ('image_dim_ordering should be "th". ' +
                                            'open ~/.keras/keras.json to change it.')

    args = Namespace(tf_type='melgram',  # which time-frequency to use
                     normalize='no', decibel=True, fmin=0.0, fmax=6000,  # mel-spectrogram params
                     n_mels=96, trainable_fb=False, trainable_kernel=False,  # mel-spectrogram params
                     conv_until=conv_until)  # how many conv layer to use? set it 4 if tagging.
    # set in [0, 1, 2, 3, 4] if feature extracting.

    model = build_convnet_model(args=args, last_layer=last_layer)
    model.load_weights('compact_cnn/weights_layer{}_{}.hdf5'.format(conv_until, K._backend),
                       by_name=True)
    # and use it!
    return model


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet

    Args:
        base_model: keras model excluding top
        nb_classes: # of classes

    Returns:
        new keras model with last layer
    """
    x = base_model.output
    predictions = Dense(10, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    plt.savefig('accuracy.png')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.savefig('loss.png')


def load_audio(audio_path, save_path):
    '''
    Load audio
    :param audio_path:
    :return:
    '''
    # mel-spectrogram parameters
    SR = 12000
    DURA = 29  # to make it 1366 frame..

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample - n_sample_fit) / 2:(n_sample + n_sample_fit) / 2]
    return src

import threading

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

class DataGenerator(object):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=32, dim=(348000, 1), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 1, 348000))
        y = np.empty((self.batch_size, 10), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, 0,] = load_audio(os.path.join("../music_genre/train/", ID), None)

            # Store class
            y_ = keras.utils.np_utils.to_categorical(self.labels[self.labels.file == ID].genre_code.values - 1,
                                                     nb_classes=10)
            y[i] = y_
        print(X.shape)
        return X, y

    @threadsafe_generator
    def get_data(self):
        while True:
            for i in range(self.__len__()):  # 1875 * 32 = 60000 -> # of training samples
                yield self.__getitem__(i)

def train_model(df_train, df_valid, args):
    base_model = load_model("feature", 3)
    nb_classes = 10
    model = add_new_last_layer(base_model, nb_classes)

    # transfer learning
    setup_to_transfer_learn(model, base_model)

    # Parameters
    params = {'dim': (348000, 1),
              'batch_size': 10,
              'n_classes': 10,
              'n_channels': 1,
              'shuffle': True}

    # Generators
    training_generator = DataGenerator(df_train['file'], df_train, **params)
    validation_generator = DataGenerator(df_valid['file'], df_valid, **params)
    model.summary()
    history_tl = model.fit_generator(
        training_generator.get_data(),
        samples_per_epoch=10, nb_epoch=20)

    # model.save(args.output_model_file)
    #if args.plot:
    #    plot_training(history_tl)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', action='store', dest='output_model_file',
                        help='Start date of training')
    parser.add_argument('-p', action='store', dest='plot',
                        help='Start date of training')
    args = parser.parse_args()
    data_folder = "data"

    df_data = pd.read_csv(os.path.join(data_folder, "train.csv"), header=None)
    df_test = pd.read_csv(os.path.join(data_folder, "test.csv"), header=None)

    df_data.columns = ["file", "genre_code"]

    len_valid = int(len(df_data) * 0.1)
    df_train = df_data.loc[: len_valid]
    df_valid = df_data.loc[len_valid : ]

    train_model(df_train, df_valid, args)
