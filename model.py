import numpy as np
import tensorflow as tf
from tensorflow import keras
from pyspin.spin import make_spin, Default


# from keras.utils.vis_utils import plot_model

# Function for creating and compiling the model, required for KerasClassifier
@make_spin(Default, "Creating the Model...")
def create_model(init_mode, dropout_rate):
    m = keras.Sequential()
    m.add(keras.layers.Dense(39, input_shape=(39,), kernel_initializer=init_mode, activation='relu'))
    m.add(keras.layers.Dense(20, activation='relu'))
    m.add(keras.layers.Dropout(dropout_rate))
    # m.add(keras.layers.Dense(10, activation='relu'))
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
