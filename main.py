from numpy import loadtxt
import pandas as pd
from tensorflow import keras
from pyspin.spin import make_spin, Default
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Loading Function
@make_spin(Default, "Loading the Dataset...")
def load_data(path):
    print('Path: ', path)
    ds = loadtxt('data/Output.csv', delimiter=',')
    output = ds[:, 39]
    input = ds[:, 0:39]
    # split into input (X) and output (y) variables
    x_train, x_test, y_train, y_test = train_test_split(input, output, test_size=0.2)
    return x_train, x_test, y_train, y_test


@make_spin(Default, "Defining the Model...")
def define_model():
    # Multi-Classification Model:
    # Die Wahrscheinlichkeit für 0 (kein Gaze Hit) und 1 (Gaze Hit) werden später in einer regressiven Form
    # wiedergegeben (Wert zwischen 0 und 1)
    m = keras.Sequential()
    m.add(keras.layers.Dense(35, input_shape=(39,), activation='relu'))
    # m.add(keras.layers.Dense(25, activation='relu'))
    m.add(keras.layers.Dense(15, activation='relu'))
    # m.add(keras.layers.Dense(8, activation='relu'))
    m.add(keras.layers.Dense(1, activation='softmax'))
    return m


@make_spin(Default, "Fitting the Model...")
def fit_model(m, X, Y):
    m.fit(X, Y, epochs=150, batch_size=1, verbose=0)
    return m


# -- Loading the Data --
p = 'data/Output.csv'
x_train, x_test, y_train, y_test = load_data(p)

# -- Defining the keras Model --
model = define_model()

# -- Compiling the keras Model --
print('Compiling the Model')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# --Fitting the Model on the Dataset--
print('Fitting the Model')
history = fit_model(model, x_train, y_train)

# --Evaluate the keras Model--
_, accuracy = model.evaluate(x_test, x_test)
print('Accuracy: %.2f' % (accuracy * 100))

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
