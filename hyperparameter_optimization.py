import numpy as np
import datetime
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from model import create_model
import process_data
import tensorflow as tf
import itertools
from itertools import permutations
# from tensorboard.plugins.hparams import api as hp

INPUT_PATH = 'data/Output.csv'
INPUT_SHAPE = 55
SEED = 3
BATCH_SIZE = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
EPOCHS = [10, 50, 100, 200, 300, 400, 500]
NEURONS_1 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
NEURONS_2 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
NEURONS_3 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
NEURONS_4 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
NEURONS_5 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
NEURONS_6 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
SMOTE = [0.1, 0.2, 0.3]
UNDER_SAMPLER = [0.1, 0.2, 0.3]
RESAMPLING = []
INIT_MODE = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
VERBOSE = 2
N_JOBS = -1
CV = 3
DATE = datetime.datetime.now().strftime('%y%m%d%H%M%S')
results = []

# fix random seed for reproducibility
np.random.seed(SEED)

# load dataset and split it to 30%
data = np.loadtxt(INPUT_PATH, delimiter=",")
split = int(len(data) * 0.3)
dataset = data[0:split]

# SMOTE and UNDERSAMPLER PAIRS watching the ratio
for s in SMOTE:
    for u in UNDER_SAMPLER:
        if u >= s:
            RESAMPLING.append([s, u])

print(len(RESAMPLING), ' Combinations of SMOTE and UNDERSAMPLER Rate')

# split into input (X) and output (Y) variables
for comb in RESAMPLING:
    print(comb)
    x_train, x_test, y_train, y_test = process_data.load_data(
                                                    INPUT_PATH,
                                                    SEED,
                                                    True,
                                                    comb[0],
                                                    comb[1]
                                                    )

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
        model__neurons_1=NEURONS_1,
        model__neurons_2=NEURONS_2,
        model__neurons_3=NEURONS_3,
        model__neurons_4=NEURONS_4,
        model__neurons_5=NEURONS_5,
        model__neurons_6=NEURONS_6,
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        n_jobs=N_JOBS,
        cv=CV
    )

    grid_result = grid.fit(x_train, y_train, callbacks=[tensorboard_callback])
    results.append(grid_result)


# summarize results
for grid_result in results:
    print(f"Best: {grid_result.best_score_:f} using {grid_result.best_params_}")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    params.append('SMOTE:{}'.format(SMOTE))
    params.append('UNDER_SAMPLER:{}'.format(UNDER_SAMPLER))
    for mean, stdev, param in zip(means, stds, params):
        print(f"{mean:f} ({stdev:f}) with: {param!r}")
