import numpy as np
import datetime
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from model import create_model
import process_data
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

INPUT_PATH = 'data/Output.csv'
SEED = 3
BATCH_SIZE = [80, 100]
EPOCHS = [500]
NEURONS_1 = [10, 20, 30, 39, 40, 41, 45, 50]
NEURONS_2 = [10, 15, 20, 25, 30, 35, 39, 40, 41, 45, 50]
INIT_MODE = ['lecun_uniform']
VERBOSE = 2
N_JOBS = -1
CV = 3
DATE = datetime.datetime.now().strftime('%y%m%d%H%M%S')

# fix random seed for reproducibility
np.random.seed(SEED)

# load dataset and split it to 30%
data = np.loadtxt(INPUT_PATH, delimiter=",")
split = int(len(data) * 0.3)
dataset = data[0:split]

# split into input (X) and output (Y) variables
x_train, x_test, y_train, y_test = process_data.load_data(INPUT_PATH, SEED, test_split=True)

# create model
model = KerasClassifier(model=create_model, verbose=VERBOSE)

# include Tensorboard History
log_path = "./hyperparam_logs/{}".format(DATE)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)

# define the grid search parameters
param_grid = dict(
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                model__init_mode=INIT_MODE,
                )

grid = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                n_jobs=N_JOBS,
                cv=CV
                )

grid_result = grid.fit(x_train, y_train, callbacks=[tensorboard_callback])

# summarize results
print(f"Best: {grid_result.best_score_:f} using {grid_result.best_params_}")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f"{mean:f} ({stdev:f}) with: {param!r}")
