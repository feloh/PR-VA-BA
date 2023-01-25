import datetime
import model
import process_data
import tensorflow as tf
import analytics

EPOCHS = 10000
BATCH_SIZE = 80
SEED = 3
VERBOSE = 2
INIT_MODE = 'lecun_uniform'
INPUT_PATH = 'data/Output.csv'
OUTPUT_PATH = 'models'
DATE = datetime.datetime.now().strftime('%y%m%d%H%M%S')
path = OUTPUT_PATH + '/' + DATE

# -- Loading the Data --
x_train, x_test, y_train, y_test = process_data.load_data(INPUT_PATH, SEED, False)

# -- Defining and compiling the keras Model --
MODEL = model.create_model(INIT_MODE)

# # --Fitting the Model on the Dataset and evaluate on Tensorboard--
log_path = "./logs/{}".format(DATE)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)

print('Fitting the Model')
hist = model.fit_model(MODEL, x_train, y_train, EPOCHS, BATCH_SIZE, VERBOSE, x_test, y_test, tensorboard_callback)

analytics.save_model(MODEL, path, EPOCHS, BATCH_SIZE)

# TODO ReadMe
# TODO Emotionen in späteren Layern erst hinzufügen
