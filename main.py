import neuralnetwork
from neuralnetwork import layers
from neuralnetwork import activations

import numpy as np


def test1():
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
    
    sum = 0

    for x in [wf, wi, wc, wo, bf, bi, bc, bo, uf, ui, uc, uo]:
        print(x.size)
        sum += x.size

    print("SUMMM : " + str(sum))

    layer = layers.LSTM(1, input_shape=(2, 2))
    layer.set_w(wf, wi, wc, wo, bf, bi, bc, bo)
    layer.set_u(uf, ui, uc, uo)
    layer._init_cp()
    layer._init_hp()
    layer.forward_propagation(_x)

# U => cell x feature
# W => cell x cell

# U => 1 x 2
# W => 1 x 1

# (m+n+1)*4*n
# (1+1+1)*4*1

def test2():
    _x = np.array([[1, 2], [.5, 3]])
    layer = layers.LSTM(1, input_shape=(2, 2))
    layer.init_layer()
    print("SIMMM : " + str(layer.count_params()))
    # layer.forward_propagation(_x)


if __name__ == "__main__":
    test1()
    test2()
