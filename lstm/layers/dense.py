import numpy as np
import copy
from lstm.activations import (
    ReLU,
    Sigmoid,
    Softmax,
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


class Utils:
    def dE_dnet(loss, y):
        _loss = copy.copy(loss)
        _loss[y] = -(1 - _loss[y])
        return _loss

    def ReLU_X(X):
        res = copy.copy(X)
        for x in np.nditer(res, op_flags=['readwrite']):
            x[...] = 1 if x > 0 else 0
        return res


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

        if activation not in [ReLU, Sigmoid, Softmax]:
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

    def backward_propagation(self, chain_matrix=None, z=0, target=0):
        if z == 1:
            # de_do = de_do
            # do_dx = np.array(self._neurons) * np.array(
            #     [(1 if i == target else 0) - n for i, n in enumerate(self._neurons)])
            # de_dx = de_do * do_dx
            # dx_dw = [1] + self._input_neurons[1:]
            # de_dw = np.array([de_dx * x for x in dx_dw])
            # self._dE_do = np.dot(de_dx, self._weights)
            # self._dE_dw = de_dw

            loss = np.array(list(map(lambda x: [x], self._neurons)))
            self._dE_do = Utils.dE_dnet(loss,
                                        target).transpose().dot(self._weights)
            self._dE_dw = Utils.dE_dnet(loss,
                                        target).dot([self._input_neurons])

            # print("==> dE/do")
            # print(self._dE_do)
            # print("de_do")
            # print(de_do)
            # print("do_dx")
            # print(do_dx)
            # print("de_dx")
            # print(de_dx)
            # print("dx_dw")
            # print(dx_dw)
            # print("de_dw")
            # print(de_dw)
            # print(self._dE_do)
        else:

            # de_dnet =  get from other (+1) layer
            # dnet_do = (net from other (+1) layer / o (activation) ) = W other layer
            # do_dx = activation derivative(x)
            # dx_dw = self._input_neurons

            # de_do = de_do
            # do_dx = [1] + [1 if x > 0 else 0 for x in self._nets]
            # de_dx = de_do * do_dx
            # dx_dw = [1] + self._input_neurons[1:]
            # de_dw = np.array([de_dx * x for x in dx_dw])
            # print("===> de_dx : ")
            # print(np.array(de_dx).shape)
            # print("===> weights : ")
            # print(np.array(self._weights).shape)
            # self._dE_do = np.dot(de_dx, np.transpose(self._weights))
            # self._dE_dw = de_dw

            # print("==> Chain_matrix")
            # print(chain_matrix.shape)
            # print("==> dx/do")
            # print(Utils.ReLU_X(np.array([1] + self._neurons)))
            # print("==> weights")
            # print(np.array(self._weights).shape)

            temp = chain_matrix * Utils.ReLU_X(np.array([1] + self._neurons))

            temp = np.array([temp[0][1:].tolist()])

            # print(temp.shape)

            self._dE_do = np.dot(temp, np.array(self._weights))
            self._dE_dw = np.dot(temp.transpose(),
                                 np.array([self._input_neurons]))

            # print(do_dx)
            # print(de_dx)
            # print(self._input_neurons)
            # print(de_dw)
            # print("self dedo")
            # print(de_dx)
            # print(self._weights)
            # print(self._dE_do)
