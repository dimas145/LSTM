import numpy as np

from neuralnetwork.activations import (
    ReLU,
    Sigmoid,
    Softmax, Linear,
    Linear
)

    # The basic terminology

    # U => units x features
    # W => units x units

    # timesteps is length of input sequence
    # features is length of a input (input vector representation)
    # units is amount of LSTM layer or hidden layer

    # n(h) == n(units)

class ForgetGate:
    def __init__(self, uf, xt, wf, hp, b):
        # (units, features) x (feature, 1) + (units, units) x (units, 1) + (units, 1)
        self._output = Sigmoid(np.matmul(uf, np.transpose(
            xt)) + np.matmul(wf, np.transpose(hp)) + np.transpose(b)).result

    @property
    def output(self):
        return self._output


class InputGate:
    def __init__(self, ui, uc, xt, wi, wc, hp, bi, bc):
        # (units, features) x (feature, 1) + (units, units) x (units, 1) + (units, 1)
        self.i = Sigmoid(np.matmul(ui, np.transpose(
            xt)) + np.matmul(wi, np.transpose(hp)) + np.transpose(bi)).result

        # (units, features) x (feature, 1) + (units, units) x (units, 1) + (units, 1)
        self.c = np.tanh(np.matmul(uc, np.transpose(
            xt)) + np.matmul(wc, np.transpose(hp)) + np.transpose(bc))


class CellState:
    def __init__(self, ft, cp, it, c_tilde):
        # (units, 1) . (units, 1) + (units, 1) . (units, 1) 
        self._output = np.multiply(ft, np.transpose(cp)) + np.multiply(it, c_tilde)

    @property
    def output(self):
        return self._output


class OutputGate:
    def __init__(self, u, x, w, ht, b, ct):
        # (units, features) x (feature, 1) + (units, units) x (units, 1) + 
        self._o = Sigmoid(np.matmul(u, np.transpose(
            x)) + np.matmul(w, np.transpose(ht)) + np.transpose(b)).result

        # (units, 1) . (units, 1)
        self._output = np.multiply(self._o, np.tanh(ct))

    @property
    def output(self):
        return self._output


class LSTM:
    def __init__(self, units, input_shape, activation=Sigmoid, recurrent_activation='sigmoid', name="lstm"):
        self._units = units
        self._timesteps = input_shape[0]
        self._features = input_shape[1]
        self._activation = activation
        self._name = name
        self.output_size = units
        self._output_size = units
        self.name = name

        self.neurons = None

        if activation not in [ReLU, Sigmoid, Softmax, Linear]:
            raise Exception("Undefined activation")
        self._activation = activation


    def init_layer(self):
        self._init_w()
        self._init_u()
        self._init_cp()
        self._init_hp()

    def count_params(self):
        sum = 0
        print("Unit: " + str(self._units))
        print("Feature: " + str(self._features))

        for x in [self._wf, self._wi, self._wc, self._wo, self._bf, self._bi, self._bc, self._bo, self._uf, self._ui, self._uc, self._uo]:
            print(x.size)
            sum += x.size

        return sum

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
        self._uf = np.random.rand(self._units, self._features)
        self._ui = np.random.rand(self._units, self._features)
        self._uc = np.random.rand(self._units, self._features)
        self._uo = np.random.rand(self._units, self._features)

    def set_w(self, wf, wi, wc, wo, bf, bi, bc, bo):
        self._wf = wf
        self._wi = wi
        self._wc = wc
        self._wo = wo
        self._bf = bf
        self._bi = bi
        self._bc = bc
        self._bo = bo

    def set_u(self, uf, ui, uc, uo):
        self._uf = uf
        self._ui = ui
        self._uc = uc
        self._uo = uo


    def _init_hp(self):
        self._hp = np.ones(shape=(1, self._units))

    def set_hp(self, hp):
        self._hp = hp

    def _init_cp(self):
        self._cp = np.ones(shape=(1, self._units))

    def set_cp(self, cp):
        self._cp = cp

    # The basic terminology

    # U => units x features
    # W => units x units

    # timesteps is length of input sequence
    # features is length of a input (input vector representation)
    # units is amount of LSTM layer or hidden layer

    def forward_propagation(self, neurons, debug=False):
        self._input_neurons = neurons
        self._init_cp()
        self._init_hp()

        for i in range(self._timesteps):
            if debug : print("Timestep", i + 1)

            x = self._input_neurons[i] # (timesteps, 1, features) 
            
            fg = ForgetGate(self._uf, x, self._wf, self._hp, self._bf)
            ft = fg.output # (units, 1)
            if debug : print("ft\t:", ft)

            ig = InputGate(self._ui, self._uc, x, self._wi,
                           self._wc, self._hp, self._bi, self._bc)
            it = ig.i # (units, 1)
            c_tilde = ig.c # (units, 1)
            if debug : print("it\t:", it)
            if debug : print("~ct\t:", c_tilde)

            cs = CellState(ft, self._cp, it, c_tilde)
            ct = cs.output # (units, 1)
            if debug : print("Ct\t:", ct)

            og = OutputGate(self._uo, x, self._wo, self._hp, self._bo, ct)
            ot = og._o
            ht = og.output
            if debug : print("ot\t:", ot)
            if debug : print("ht\t:", ht)
            self._cp = ct
            self._hp = ht
            self.neurons = np.transpose(self._hp)[0].tolist()
            if debug : print()
            
