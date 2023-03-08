import datetime
import model
import process_data
import tensorflow as tf
import analytics

EPOCHS = 10000
BATCH_SIZE = 80
INIT_MODE = 'lecun_uniform'
INPUT_SHAPE = 55
NEURONS_1 = 30
NEURONS_2 = 50
NEURONS_3 = 50
NEURONS_4 = 50
NEURONS_5 = 50
NEURONS_6 = 50
SMOTE = 0.2
UNDER_SAMPLER = 0.2
SEED = 3
VERBOSE = 2
INPUT_PATH = 'data/Output.csv'
OUTPUT_PATH = 'models'
DATE = datetime.datetime.now().strftime('%y%m%d%H%M%S')
path = OUTPUT_PATH + '/' + DATE

# -- Loading the Data --
x_train, x_test, y_train, y_test = process_data.load_data(INPUT_PATH, SEED, False, SMOTE, UNDER_SAMPLER)

# -- Defining and compiling the keras Model --
MODEL = model.create_model(INIT_MODE, INPUT_SHAPE, NEURONS_1, NEURONS_2, NEURONS_3, NEURONS_4, NEURONS_5, NEURONS_6)

# # --Fitting the Model on the Dataset and evaluate on Tensorboard--
log_path = "./logs/{}".format(DATE)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)

print('Fitting the Model')
hist = model.fit_model(MODEL, x_train, y_train, EPOCHS, BATCH_SIZE, VERBOSE, x_test, y_test, tensorboard_callback)

analytics.save_model(MODEL, path, EPOCHS, BATCH_SIZE, hist)

