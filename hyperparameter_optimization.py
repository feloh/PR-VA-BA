import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from model import create_model

INPUT_PATH = 'data/Output.csv'
SEED = 7
BATCH_SIZE = [10, 20, 40, 60, 80, 100]
EPOCHS = [10, 50, 100]

# fix random seed for reproducibility
tf.random.set_seed(SEED)

# load dataset and split it to 30%
data = np.loadtxt(INPUT_PATH, delimiter=",")
split = int(len(data)*0.3)
dataset = data[0:split]

# split into input (X) and output (Y) variables
X = dataset[:, 0:39]
Y = dataset[:, 39]

# create model
model = KerasClassifier(model=create_model, verbose=0)

# define the grid search parameters
param_grid = dict(batch_size=BATCH_SIZE, epochs=EPOCHS, )
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
