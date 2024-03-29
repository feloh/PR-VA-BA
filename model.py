import numpy as np
import tensorflow as tf
from tensorflow import keras
from pyspin.spin import make_spin, Default


# from keras.utils.vis_utils import plot_model

# Function for creating and compiling the model, required for KerasClassifier
@make_spin(Default, "Creating the Model...")
def create_model(init_mode, input_shape, neurons_1, neurons_2, neurons_3, neurons_4, neurons_5, neurons_6):
    m = keras.Sequential()
    m.add(keras.layers.Dense(neurons_1,
                             input_shape=(input_shape,),
                             kernel_initializer=init_mode,
                             activation='relu',
                             ))

    m.add(keras.layers.Dense(neurons_2, activation='relu'))
    m.add(keras.layers.Dense(neurons_3, activation='relu'))
    m.add(keras.layers.Dense(neurons_4, activation='relu'))
    m.add(keras.layers.Dense(neurons_5, activation='relu'))
    m.add(keras.layers.Dense(neurons_6, activation='relu'))
    m.add(keras.layers.Dense(1, activation='sigmoid'))
    # Compile model
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m


# Function for fitting the model
def fit_model(m, X, y, e, bs, v, xv, yv, tb_c):
    m.fit(X, y, epochs=e, batch_size=bs, verbose=v, validation_data=(xv, yv), callbacks=[tb_c])
    return m


# Function to plot the Sequential Model Structure
def plot_model(m, path):
    keras.utils.plot_model(m, to_file=path, show_layer_names=True)
