import numpy as np
import os
import datetime
from tensorflow import keras
from pyspin.spin import make_spin, Default
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import numpy


# Loading Function, Spliting the Data into Test and Train Data, Using SMOTE and RandomUndersampler to balance the Data
@make_spin(Default, "Loading the Dataset...")
def load_data(path):
    print('Path: ', path)
    ds = np.loadtxt('data/Output.csv', delimiter=',')
    seed = 7
    numpy.random.seed(seed)
    input = ds[:, 0:39]
    output = ds[:, 39]
    x_train, x_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=seed)

    # define pipeline
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    # transform the dataset
    x_train, y_train = pipeline.fit_resample(x_train, y_train)

    return x_train, x_test, y_train, y_test


@make_spin(Default, "Defining the Model...")
def define_model():
    m = keras.Sequential()
    m.add(keras.layers.Dense(39, input_shape=(39,), activation='relu'))
    m.add(keras.layers.Dense(20, activation='relu'))
    m.add(keras.layers.Dense(10, activation='relu'))
    m.add(keras.layers.Dense(1, activation='softmax'))
    return m


def fit_model(m, X, y, e, bs, v, xv, yv):
    m.fit(X, y, epochs=e, batch_size=bs, verbose=v, validation_data=(xv, yv))
    return m


EPOCHS = 10
BATCH_SIZE = 1
VERBOSE = 2
INPUT_PATH = 'data/Output.csv'
OUTPUT_PATH = 'models'
DATE = datetime.datetime.now().strftime('%y%m%d%H%M%S')

# -- Loading the Data --
x_train, x_test, y_train, y_test = load_data(INPUT_PATH)

# -- Defining the keras Model --
model = define_model()

# -- Compiling the keras Model --
print('Compiling the Model')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # --Fitting the Model on the Dataset and evaluate--
print('Fitting the Model')
hist = fit_model(model, x_train, y_train, EPOCHS, BATCH_SIZE, VERBOSE, x_test, y_test)

# -- save Model to Tensorflow SavedModel-Format --
path = OUTPUT_PATH + '/' + DATE
print(path)
model_path = "{}/model".format(path)
model.save(model_path)

# -- serialize model to JSON --
model_json = model.to_json()
model_json_path = "{}/{}_{}_model.json".format(path, EPOCHS, BATCH_SIZE)
with open(model_json_path, "w") as json_file:
    json_file.write(model_json)

print("Saved model to disk")

# list all history params
print(hist.history.params)

# list all data in history
print(hist.history.history.keys())

# summarize history for accuracy
plt.plot(hist.history.history['accuracy'])
plt.plot(hist.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
accuracy_path = "{}/accuracy.png".format(path)
plt.savefig(accuracy_path)

# summarize history for loss
plt.plot(hist.history.history['loss'])
plt.plot(hist.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
loss_path = "{}/loss.png".format(path)
plt.savefig(loss_path)

# TODO Fix saving Plots
# TODO ReadMe
# TODO Split Functions among the Scripts (only as much as possible)
# TODO Emotionen in späteren Layern erst hinzufügen
