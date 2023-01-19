import datetime
import matplotlib.pyplot as plt
import model
import process_data

EPOCHS = 10
BATCH_SIZE = 1
SEED = 7
VERBOSE = 2
INPUT_PATH = 'data/Output.csv'
OUTPUT_PATH = 'models'
DATE = datetime.datetime.now().strftime('%y%m%d%H%M%S')

# -- Loading the Data --
x_train, x_test, y_train, y_test = process_data.load_data(INPUT_PATH, SEED)

# -- Defining and compiling the keras Model --
MODEL = model.create_model()

# # --Fitting the Model on the Dataset and evaluate--
print('Fitting the Model')
hist = model.fit_model(MODEL, x_train, y_train, EPOCHS, BATCH_SIZE, VERBOSE, x_test, y_test)

# -- save Model to Tensorflow SavedModel-Format --
path = OUTPUT_PATH + '/' + DATE
print(path)
model_path = "{}/model".format(path)
model.save(model_path)

# -- Plot Model-Structure --

model.plot_model(MODEL, path)

# -- serialize model to JSON --
model_json = MODEL.to_json()
model_json_path = "{}/{}_{}_model.json".format(path, EPOCHS, BATCH_SIZE)
with open(model_json_path, "w") as json_file:
    json_file.write(model_json)

print("Saved model to disk")

# list all history params
print(hist.history.params)

# list all data in history
print(hist.history.history.keys())

# summarize history for accuracy
acc_fig = plt.figure()
plt.plot(hist.history.history['accuracy'])
plt.plot(hist.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
accuracy_path = "{}/accuracy.png".format(path)
acc_fig.savefig(accuracy_path, dpi=acc_fig.dpi)

# summarize history for loss
lss_fig = plt.figure()
plt.plot(hist.history.history['loss'])
plt.plot(hist.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
loss_path = "{}/loss.png".format(path)
lss_fig.savefig(loss_path, dpi=lss_fig.dpi)

# TODO ReadMe
# TODO Split Functions among the Scripts (only as much as possible)
# TODO Emotionen in späteren Layern erst hinzufügen
