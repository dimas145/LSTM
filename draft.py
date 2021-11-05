import numpy as np
from neuralnetwork.activations import (
    ReLU,
    Sigmoid,
    Softmax,
)
import pandas as pd
import matplotlib.pyplot as plt


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
    def __init__(self, units, input_size=0, activation=Sigmoid, recurrent_activation='sigmoid', name="lstm"):
        self._units = units
        self._input_size = input_size
        self._activation = activation
        self._name = name

        if activation not in [ReLU, Sigmoid, Softmax]:
            raise Exception("Undefined activation")
        self._activation = activation

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
        self._uf = np.random.rand(self._units, self._input_size)
        self._ui = np.random.rand(self._units, self._input_size)
        self._uc = np.random.rand(self._units, self._input_size)
        self._uo = np.random.rand(self._units, self._input_size)
        self._buf = np.random.rand(1, self._input_size)
        self._bui = np.random.rand(1, self._input_size)
        self._buc = np.random.rand(1, self._input_size)
        self._buo = np.random.rand(1, self._input_size)

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

# hidden => jumlah cell
# input  => jumlah timestep
# output => jumlah timestep

# U => hidden x input
# V => output x hidden
# W => hidden x hidden

# U => 1 x 2
# V => 2 x 1
# W => 1 x 1

# slide
_x = np.array([[1, 2], [.5, 3]])
uf = np.array([.7, .45])
ui = np.array([.95, .8])
uc = np.array([.45, .25])
uo = np.array([.6, .4])

wf = np.array([.1])
wi = np.array([.8])
wc = np.array([.15])
wo = np.array([.25])

bf = np.array([.15])
bi = np.array([.65])
bc = np.array([.2])
bo = np.array([.1])

hp = np.array([0])
cp = np.array([0])

buf = np.array([0])
bui = np.array([0])
buc = np.array([0])
buo = np.array([0])

layer = LSTM(2)
layer.set_w(wf, wi, wc, wo, bf, bi, bc, bo)
layer.set_u(uf, ui, uc, uo, buf, bui, buc, buo)
layer.forward_propagation(_x)

print()

layer = LSTM(10, input_size=50)

print(4*layer._units*(5+layer._units+1))

# bitcoin price
# df = pd.read_csv('../bitcoin_price_Training - Training.csv')
# print(df.head())

# df = df[::-1]
# print(df[::-1].drop(['Date', 'Volume'], axis=1).to_numpy())
# high_df = df[::-1][['Date', 'High']]
# low_df = df[::-1][['Date', 'Low']]
# open_df = df[::-1][['Date', 'Open']]
# close_df = df[::-1][['Date', 'Close']]

# fig, axes = plt.subplots(2, 2)
# df.plot.line('Date', 'High', ax=axes[0, 0])
# df.plot.line('Date', 'Low', ax=axes[0, 1])
# df.plot.line('Date', 'Open', ax=axes[1, 0])
# df.plot.line('Date', 'Close', ax=axes[1, 1])
# plt.show()
