# load and evaluate a saved model
from numpy import loadtxt
from tensorflow import keras

INPUT_PATH = 'data/Output.csv'
model_number = input('Enter the Model-Number:')
MODEL_PATH = 'models/{}/model'.format(model_number)
# load model
model = keras.models.load_model(MODEL_PATH)
# summarize model.
model.summary()
# load dataset
dataset = loadtxt(INPUT_PATH, delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:39]
Y = dataset[:, 39]
# evaluate the model
score = model.evaluate(X, Y, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))