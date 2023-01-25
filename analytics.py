import matplotlib.pyplot as plt


def save_model(MODEL, path, EPOCHS, BATCH_SIZE, hist):
    print("Saved model to disk")
    # -- serialize model to JSON --
    model_json = MODEL.to_json()
    model_json_path = "{}/{}_{}_model.json".format(path, EPOCHS, BATCH_SIZE)
    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)

    # -- save Model to Tensorflow SavedModel-Format --
    model_path = "{}/model".format(path)
    MODEL.save(model_path)

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
