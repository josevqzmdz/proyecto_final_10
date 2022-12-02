from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import CNN_MNIST.entrenamiento as CNN
import CNN_MNIST.informacion as info

import CNN_MNIST.informacion as info

class entrenamiento:
    def __init__(self):
        model = CNN()

        # Define o optimizador
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

        # Compila o model
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        # Set a learning rate annealer
        # learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

        model_checkpoint = ModelCheckpoint('digitos.hdf5', monitor='loss', save_best_only=True)

        epochs = 1  # Turn epochs to 30 to get 0.9967 accuracy
        batch_size = 86

        # entrenadores de info
        informacion = info()
        X_train, Y_train = informacion.load_train_data()
        X_val, Y_val = informacion.load_test_data()

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        datagen.fit(X_train)

        # Fit the model
        history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), epochs=epochs,
                                      validation_data=(X_val, Y_val),
                                      verbose=1, steps_per_epoch=X_train.shape[0] // batch_size,
                                      callbacks=[model_checkpoint])

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(history.history['loss'], color='b', label="Training loss")
        ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
        legend = ax[0].legend(loc='best', shadow=True)

        ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
        ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
        legend = ax[1].legend(loc='best', shadow=True)
        plt.show()

    def CNN(weights_path=None):

        # Set the CNN model
        # my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation="softmax"))

        if weights_path:
            model.load_weights(weights_path)

        return model


