
# fix "no module named..." in conda
# https://www.youtube.com/watch?v=I9st-DgQoWc
from MNIST_Filtrado.Filtrado import Filtrado as filtrado
from MNIST_Filtrado.Neural_Net import Neural_Net as nn

if __name__ == '__main__':
    filtrado()
    nn()
