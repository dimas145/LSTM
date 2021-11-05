import neuralnetwork
from neuralnetwork import layers
from neuralnetwork import activations

import numpy as np


def test1():

    model1 = neuralnetwork.Sequential()

    model1.add(layers.LSTM(1, input_shape=(2, 2)))
    model1.add(layers.Dense(1, activation=activations.ReLU))

    model1.summary()


def test2():
    model2 = neuralnetwork.Sequential()

    model2.add(layers.LSTM(10, input_shape=(32, 6)))
    model2.add(layers.Dense(1, activation=activations.ReLU))

    model2.summary()

def test3():
    _x = np.array([[[1, 2]], [[.5, 3]]])
    uf = np.array([[.7, .45]])
    ui = np.array([[.95, .8]])
    uc = np.array([[.45, .25]])
    uo = np.array([[.6, .4]])

    wf = np.array([[.1]])
    wi = np.array([[.8]])
    wc = np.array([[.15]])
    wo = np.array([[.25]])

    bf = np.array([[.15]])
    bi = np.array([[.65]])
    bc = np.array([[.2]])
    bo = np.array([[.1]])

    layer1 = layers.LSTM(1, input_shape=(2, 2))

    layer1.set_w(wf, wi, wc, wo, bf, bi, bc, bo)
    layer1.set_u(uf, ui, uc, uo)

    layer1._init_cp()
    layer1._init_hp()

    layer1.forward_propagation(_x, debug=True)

def test4():
    print()
    print("=> Test dji dengan dataset multifeatures 32 timesteps dapat dilihat di notebook main.ipynb")
    print("=> Pengujian dilakukan dengan menginisiasi weights model tubes dengan weights training LSTM keras, namun ternyata nilai output gates model keras tidak dapat diambil. Proses menghasilkan nilai dengan dasar output gates yang berbeda.")


if __name__ == "__main__":

    print()
    print("# Notes: pengujian dan percobaan lebih lengkap terdapat dalam notebook main.ipynb!\n\n")

    print("=> Model from IF4071 Lecture Slide")
    test1()

    print()

    print("=> Model with 32 Timesteps, 6 Features, and 10 Units LSTM Random")
    test2()

    print()

    print("=> IF4071 Lecture Forward Propagation Example")
    test3()

    print()

    print("=> Test uji data timeseries bitcoin")
    test4()
    
