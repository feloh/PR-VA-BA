import numpy as np
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from model import create_model
import process_data

INPUT_PATH = 'data/Output.csv'
SEED = 7
BATCH_SIZE = [1, 5, 10, 20, 40, 60, 80, 100, 1000]
EPOCHS = [1000, 5000, 10000]
VERBOSE = 0
N_JOBS = -1
CV = 3

# fix random seed for reproducibility
np.random.seed(SEED)

# load dataset and split it to 30%
data = np.loadtxt(INPUT_PATH, delimiter=",")
split = int(len(data)*0.3)
dataset = data[0:split]

# split into input (X) and output (Y) variables
x_train, x_test, y_train, y_test = process_data.load_data(INPUT_PATH, SEED, test_split=True)

# create model
model = KerasClassifier(model=create_model, verbose=VERBOSE)

# define the grid search parameters
param_grid = dict(batch_size=BATCH_SIZE, epochs=EPOCHS, )
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=N_JOBS, cv=CV)
grid_result = grid.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
