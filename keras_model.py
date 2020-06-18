#
# The SELDnet architecture
#

import tensorflow as tf

#from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input
Bidirectional = tf.keras.layers.Bidirectional
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Input = tf.keras.layers.Input
#from tensorflow.keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
Dense = tf.keras.layers.Dense
Activation = tf.keras.layers.Activation
Dropout = tf.keras.layers.Dropout
Reshape = tf.keras.layers.Reshape
Permute = tf.keras.layers.Permute
#from keras.layers.recurrent import GRU
GRU = tf.keras.layers.GRU
#from keras.layers.normalization import BatchNormalization
BatchNormalization = tf.keras.layers.BatchNormalization
#from keras.models import Model
Model = tf.keras.models.Model
#from keras.layers.wrappers import TimeDistributed
TimeDistributed = tf.keras.layers.TimeDistributed
#from keras.optimizers import Adam
Adam = tf.keras.optimizers.Adam
#import keras
keras = tf.keras
keras.backend.set_image_data_format('channels_first')
from IPython import embed


def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size,
                                rnn_size, fnn_size, weights):
    # model definition
    spec_start = Input(shape=(data_in[-3], data_in[-2], data_in[-1]))

    # CNN
    spec_cnn = spec_start
    # spec_cnn = keras.Sequential([spec_start])
    for i, convCnt in enumerate(pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        # spec_cnn.add(Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same'))
        spec_cnn = BatchNormalization()(spec_cnn)
        # spec_cnn.add(BatchNormalization())
        spec_cnn = Activation('relu')(spec_cnn)
        # spec_cnn.add(Activation('relu'))
        spec_cnn = MaxPooling2D(pool_size=(1, pool_size[i]))(spec_cnn)
        # spec_cnn.add(MaxPooling2D(pool_size=(1, pool_size[i])))
        spec_cnn = Dropout(dropout_rate)(spec_cnn)
        # spec_cnn.add(Dropout(dropout_rate))
    spec_cnn = Permute((2, 1, 3))(spec_cnn)
    # spec_cnn.add(Permute((2, 1, 3)))

    # RNN
    spec_rnn = Reshape((data_in[-2], -1))(spec_cnn)
    # spec_rnn = keras.models.clone_model(spec_cnn)
    # spec_rnn.add(Reshape((data_in[-2], -1)))
    # print("----shape={}".format(spec_cnn.shape))
    # print("----rnn_size={}".format(rnn_size))#----nb_rnn_filt={}\n
    # spec_cnn.summary()
    # print("----spec_rnn.output_shape={}".format(spec_rnn.output_shape))
    for nb_rnn_filt in rnn_size:
        # spec_rnn 
        bi = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True),
            merge_mode='mul'
        )(spec_rnn)
        # bi.summary()
        # print("----bi.shape={}".format(bi.shape))
        spec_rnn.add(bi)

    # FC - DOA
    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)

    doa = TimeDistributed(Dense(data_out[1][-1]))(doa)
    doa = Activation('linear', name='doa_out')(doa)

    # FC - SED
    sed = spec_rnn
    for nb_fnn_filt in fnn_size:
        sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    model = Model(inputs=spec_start, outputs=[sed, doa])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)

    model.summary()
    # doa = keras.Sequential([
    #     Input(shape=(data_in[-3], data_in[-2], data_in[-1]))
    #     keras.layers.Flatten(input_shape=(data_in[-3], data_in[-2], data_in[-1])),
    #     keras.layers.Dense(128, activation='relu'),
    #     keras.layers.Dense(10)
    # ])
    # doa = TimeDistributed(Dense(data_out[1][-1]))(doa)
    # doa = Activation('linear', name='doa_out')(doa)

    # sed = keras.Sequential([
    #     keras.layers.Flatten(input_shape=(data_in[-3], data_in[-2], data_in[-1])),
    #     keras.layers.Dense(128, activation='relu'),
    #     keras.layers.Dense(10)
    # ])
    # sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
    # sed = Activation('sigmoid', name='sed_out')(sed)

    # model = Model(inputs=spec_start, outputs=[sed, doa])
    # model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)
    return model
