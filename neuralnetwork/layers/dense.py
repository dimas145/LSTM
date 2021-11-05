import numpy as np
import copy
from neuralnetwork.activations import (
    ReLU,
    Sigmoid,
    Softmax,
    Linear
)


class Matrix:
    def add(mat1, mat2):
        mat1 = np.array(mat1)
        mat2 = np.array(mat2)

        return np.add(mat1, mat2).tolist()

    def mult(mat1, mat2):
        mat1 = np.array(mat1)
        mat2 = np.array(mat2)

        return np.dot(mat1, mat2).tolist()


class Dense:
    def __init__(
        self,
        output_size,
        name="dense",
        activation=None,
        input_size=10,
    ):
        self._name = name
        self._output_size = output_size
        self._output_shape = (None, output_size)
        self._input_size = input_size

        if activation not in [ReLU, Sigmoid, Softmax, Linear]:
            raise Exception("Undefined activation")
        self._activation = activation

        self._input_neurons = [-1] * (self._input_size + 1)
        self._neurons = [-1] * (self._output_size + 1)
        self._weights = []

        self._dE_dw = None

    @property
    def activation(self):
        return self._activation

    @property
    def output_size(self):
        return self._output_size

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, input_size):
        self._input_size = input_size

    @property
    def size(self):
        return self._output_size

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def neurons(self):
        return self._neurons

    def init_layer(self):
        self.init_weights()

    def init_weights(self):
        limit = np.sqrt(1 / float(self._input_size))
        self._weights = np.random.normal(0.0,
                                         limit,
                                         size=(self._output_size,
                                               self._input_size)).tolist()
        bias_weight = np.random.normal(0.0, limit)

        for i in range(len(self._weights)):
            self._weights[i].insert(0, bias_weight)

    def set_weights(self, weights):
        self._weights = weights

    def set_outputs_value_by_matrix(self, hk):
        self._input_neurons = hk

    def forward_propagation(self, neurons, y=0):
        self._input_neurons = neurons

        neurons = list(map(lambda x: [x], neurons))

        ak = list(map(lambda x: x[0], Matrix.mult(self._weights, neurons)))
        hk = self._activation(ak).result

        self._nets = ak
        self._neurons = hk
