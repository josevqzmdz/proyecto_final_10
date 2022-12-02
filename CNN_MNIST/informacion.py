import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

class informacion:
    def __init__(self):

        np.random.seed(2)
        sns.set(style='white', context='notebook', palette='deep')
        self.create_train_data()

        # destructor
        del self.train

        # exibe una cantidad
        self.Y_train.value_counts()
        g = sns.countplot(self.Y_train)
        plt.show()

        #  verificando os valores nulos e ausentes
        self.X_train.isnull().any().describe()

        # Normaliza os dados
        X_train = self.normalizacao(self.X_train)

        # muda a forma em 3 dimensoes
        X_train = X_train.values.reshape(-1, 28, 28, 1)  # (42000, 28, 28, 1)

        # Codificaçao de rotulo
        Y_train = to_categorical(self.Y_train, num_classes=10)

        g = plt.imshow(X_train[1][:, :, 0])
        plt.show()

        # Set the random seed
        random_seed = 2

        # divide os dados de treino e validacao para setar no treinamento
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)

        np.save('/home/nig/PycharmProjects/DL-Introduction-CNN/MNIST/data/npy/X_train.npy', X_train)
        np.save('/home/nig/PycharmProjects/DL-Introduction-CNN/MNIST/data/npy/X_test.npy', X_test)
        np.save('/home/nig/PycharmProjects/DL-Introduction-CNN/MNIST/data/npy/Y_train.npy', Y_train)
        np.save('/home/nig/PycharmProjects/DL-Introduction-CNN/MNIST/data/npy/Y_test.npy', Y_test)


    def normalizacao(data):
        return data / 255.0


    def create_train_data(self):

        print('running create_train_data() ')
        train = pd.read_csv(os.getcwd()+"/MNIST/data/input/train.csv")

        Y_train = train["label"]

        # Drop 'label' column
        X_train = train.drop(labels=["label"], axis=1) # (42000, 784)

    def load_train_data(self):
        X_train = np.load('/home/nig/PycharmProjects/DL-Introduction-CNN/MNIST/data/npy/X_train.npy')
        Y_train = np.load('/home/nig/PycharmProjects/DL-Introduction-CNN/MNIST/data/npy/Y_train.npy')
        return X_train, Y_train

    def load_test_data(self):
        X_test = np.load('/home/nig/PycharmProjects/DL-Introduction-CNN/MNIST/data/npy/X_test.npy')
        Y_test = np.load('/home/nig/PycharmProjects/DL-Introduction-CNN/MNIST/data/npy/Y_test.npy')
        return X_test, Y_test






