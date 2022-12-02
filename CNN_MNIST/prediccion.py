from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix

import CNN_MNIST.informacion as info
import CNN_MNIST.entrenamiento as CNN
import numpy as np
from MNIST_assets.functions import plot_confusion_matrix
import CNN_MNIST.utilidad as util
from sklearn.metrics import accuracy_score

class prediccion:
    def __init__(self):

        info_td = info()
        X_test, Y_test = info_td.load_test_data()

        model = CNN('digitos.hdf5')
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        Y_pred = model.predict(X_test)

        # Convert predictions classes to one hot vectors
        Y_pred_classes = np.argmax(Y_pred, axis=1)

        # Convert validation observations to one hot vectors
        Y_true = np.argmax(Y_test, axis=1)

        # compute the confusion matrix
        confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

        print(" ACURÀCIA = %.2f" %(accuracy_score(Y_true, Y_pred_classes)*100), "%")

        # plot the confusion matrix
        plot_confusion_matrix(confusion_mtx, classes=range(10))
