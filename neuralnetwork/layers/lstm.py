import numpy as np

from neuralnetwork.activations import (
    ReLU,
    Sigmoid,
    Softmax,
)

class ForgetGate:
    def __init__(self, uf, xt, wf, hp, b):
        self._output = Sigmoid(np.matmul(uf, np.transpose(
            xt)) + np.matmul(wf, np.transpose(hp)) + b).result

    @property
    def output(self):
        return self._output


class InputGate:
    def __init__(self, ui, uc, xt, wi, wc, hp, bi, bc):
        self.i = Sigmoid(np.matmul(ui, np.transpose(
            xt)) + np.matmul(wi, np.transpose(hp)) + bi).result
        self.c = np.tanh(np.matmul(uc, np.transpose(
            xt)) + np.matmul(wc, np.transpose(hp)) + bc)


class CellState:
    def __init__(self, ft, cp, it, c_tilde):
        self._output = np.multiply(ft, cp) + np.multiply(it, c_tilde)

    @property
    def output(self):
        return self._output


class OutputGate:
    def __init__(self,  u, x, w, ht, b, ct):
        self._o = Sigmoid(np.matmul(u, np.transpose(
            x)) + np.matmul(w, np.transpose(ht)) + b).result

        self._output = np.multiply(self._o, np.tanh(ct))

    @property
    def output(self):
        return self._output


class LSTM:
    def __init__(self, units, input_shape=0, activation=Sigmoid, recurrent_activation='sigmoid', name="lstm"):
        self._units = units
        self._timestamp = input_shape[0]
        self._features = input_shape[1]
        self._activation = activation
        self._name = name
        self.output_size = units
        self._output_size = units
        self.name = name

        if activation not in [ReLU, Sigmoid, Softmax]:
            raise Exception("Undefined activation")
        self._activation = activation


    def init_layer(self):
        self._init_w()
        self._init_u()
        self._init_cp()
        self._init_hp()


    def _init_w(self):
        self._wf = np.random.rand(self._units, self._units)
        self._wi = np.random.rand(self._units, self._units)
        self._wc = np.random.rand(self._units, self._units)
        self._wo = np.random.rand(self._units, self._units)
        self._bf = np.random.rand(1, self._units)
        self._bi = np.random.rand(1, self._units)
        self._bc = np.random.rand(1, self._units)
        self._bo = np.random.rand(1, self._units)

    def _init_u(self):
        self._uf = np.random.rand(self._units, self._timestamp)
        self._ui = np.random.rand(self._units, self._timestamp)
        self._uc = np.random.rand(self._units, self._timestamp)
        self._uo = np.random.rand(self._units, self._timestamp)
        self._buf = np.random.rand(1, self._timestamp)
        self._bui = np.random.rand(1, self._timestamp)
        self._buc = np.random.rand(1, self._timestamp)
        self._buo = np.random.rand(1, self._timestamp)

    def set_w(self, wf, wi, wc, wo, bf, bi, bc, bo):
        self._wf = wf
        self._wi = wi
        self._wc = wc
        self._wo = wo
        self._bf = bf
        self._bi = bi
        self._bc = bc
        self._bo = bo

    def set_u(self, uf, ui, uc, uo, buf, bui, buc, buo):
        self._uf = uf
        self._ui = ui
        self._uc = uc
        self._uo = uo
        self._buf = buf
        self._bui = bui
        self._buc = buc
        self._buo = buo

    def _init_hp(self):
        self._hp = np.array([0])

    def set_hp(self, hp):
        self._hp = hp

    def _init_cp(self):
        self._cp = np.array([0])

    def set_cp(self, cp):
        self._cp = cp

    def forward_propagation(self, neurons, y=0):
        self._input_neurons = neurons

        for i in range(self._units):
            print("Step", i + 1)

            x = self._input_neurons[i]
            fg = ForgetGate(self._uf, x, self._wf, self._hp, self._bf)
            ft = fg.output
            print("ft:", ft)

            ig = InputGate(self._ui, self._uc, x, self._wi,
                           self._wc, self._hp, self._bi, self._bc)
            it = ig.i
            c_tilde = ig.c
            print("it:", it)
            print("c_tilde:", c_tilde)

            cs = CellState(ft, self._cp, it, c_tilde)
            ct = cs.output
            print("ct:", ct)

            og = OutputGate(self._uo, x, self._wo, self._hp, self._bo, ct)
            ot = og._o
            ht = og.output
            print("ot:", ot)
            print("ht:", ht)
            self._cp = ct
            self._hp = ht
