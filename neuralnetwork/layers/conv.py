import numpy as np
import copy
from neuralnetwork.activations import (
    ReLU,
    Sigmoid,
    Softmax,
)

class Utils:
    
    def dE_dnet(loss, y):
        _loss = copy.copy(loss)
        _loss[y] = - (1 - _loss[y])
        return _loss

    def ReLU_X(X):
        res = copy.copy(X)
        for x in np.nditer(res, op_flags=['readwrite']):
           x[...] = 1 if x > 0  else 0
        return res

    def X_ReLU(X, ReLU):
        _X = copy.copy(X)
        _ReLU = copy.copy(ReLU)
        for i in range(len(_ReLU)):
            for j in range(len(_ReLU[i])):
                for k in range(len(_ReLU[i][j])):
                    _ReLU[i][j][k] = 1 if _ReLU[i][j][k] == _X[i] else 0                    
        return _ReLU

    def constant_mult_matrix(m1, m2):
        resi = []
        for i in range(len(m2)):
            resj = []
            for j in range(len(m2[i])):
                resk = []
                for k in range(len(m2[i][j])):
                    resk.append(m2[i][j][k] * m1[i])
                resj.append(resk)
            resi.append(resj)                
                    
        return np.array(resi)

    def biases_correction(weights):
        res = []
        for i in range(len(weights)):
            sum = 0
            for j in range(len(weights[i])):
                for k in range(len(weights[i][j])):
                    sum += weights[i][j][k]
            res.append(sum)
        return res

    def convolution(matrix, weights, strides):
        height = len(matrix)
        width = len(matrix[0])

        conv = []

        for z in range(len(weights)):
            temp2 = []
            for i in range(0, height - len(weights[z]) + 1, strides[0]):
                temp1 = []
                
                for j in range(0, width - len(weights[z][i]) + 1, strides[1]):
                    sum = 0
                    for k in range(len(weights[z])):
                        for l in range(len(weights[z][i])):
                            sum += matrix[i + k][j + l] * weights[z][k][l]
                    temp1.append(sum)
                temp2.append(temp1)
            conv.append(temp2)

        return np.array(conv)

class Conv2D:
    def __init__(
            self,
            filters,
            kernel_size,
            name="conv2d",
            strides=(1, 1),
            padding=(0, 0),
            input_shape=None,
            activation=None,
    ):
        self._name = name

        if activation not in [ReLU, Sigmoid, Softmax]:
            raise Exception("Undefined activation")
        self._activation = activation

        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding

        self._input_shape = input_shape
        self._output_shape = None

        self._input_neurons = []
        
        self._nets = []

        self._neurons = []
        self._biases = []

        self._weights = []
        self._dE_dw = None
        self._dE_do = None

    
    def init_layer(self):
        self.calculate_output_spatial_size()
        self.init_weights()

    def init_weights(self):

        limit = np.sqrt(1 / float(self._input_shape[1] * self._input_shape[2] *
                                  self._input_shape[3]))

        for _ in range(self._filters):
            self._weights.append(
                np.random.normal(0.0,
                                 limit,
                                 size=(self._kernel_size[1],
                                       self._kernel_size[0])).tolist())

    def set_weights(self, weights):
        self._weights = weights

    def init_biases(self):
        for _ in range(self._filters):
            self._biases.append([0] * self._kernel_size[0])

    @property
    def output_size(self):
        return self._output_shape

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def input_neurons(self):
        return self._input_neurons

    @input_neurons.setter
    def input_neurons(self, neurons):
        self._input_neurons = neurons

    @property
    def neurons(self):
        return self._neurons

    @property
    def nets(self):
        return self._nets

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def input_size(self):
        return self._input_shape

    @input_size.setter
    def input_size(self, shape):
        self._input_shape = shape

    def set_outputs_value_by_matrix(self, convoluted, detected):
        self._nets = convoluted
        self._neurons = detected

    def calculate_output_spatial_size(self):
        if (self._input_shape[0] is not None):
            self._input_shape = (None, self._input_shape[0],
                                 self._input_shape[1], self._input_shape[2])
        W = self._input_shape[1]
        F = self._kernel_size[0]
        P = self._padding[0]
        S = self._strides[0]
        K = self._filters
        V = round(((W - F + (2 * P)) / S) + 1)

        self._output_shape = (None, V, V, K)

    def add_auto_padding(self, matrix):
        height = len(matrix)
        width = len(matrix[0])

        left_padding = 0
        right_padding = 0

        up_padding = 0
        down_padding = 0

        if (width % self._strides[1] != 0):
            for i in range(self._strides[1] - (width % self._strides[1])):
                if (i % 2 == 0):
                    right_padding += 1
                else:
                    left_padding += 1

        if (height % self._strides[0] != 0):
            for i in range(self._strides[0] - (height % self._strides[0])):
                if (i % 2 == 0):
                    down_padding += 1
                else:
                    up_padding += 1

        for i in range(height):
            matrix[i] += [0] * right_padding
            for _ in range(left_padding):
                matrix[i].insert(0, 0)

        for _ in range(up_padding):
            matrix.insert(0, [0] * len(matrix[0]))

        for _ in range(down_padding):
            matrix.append([0] * len(matrix[0]))

        return matrix

    def add_padding(self, matrix):
        height = len(matrix)

        left_padding = self._padding[1]
        right_padding = self._padding[1]

        up_padding = self._padding[0]
        down_padding = self._padding[0]

        for i in range(height):
            matrix[i] += [0] * right_padding
            for _ in range(left_padding):
                matrix[i].insert(0, 0)

        for _ in range(up_padding):
            matrix.insert(0, [0] * len(matrix[0]))

        for _ in range(down_padding):
            matrix.append([0] * len(matrix[0]))

        return matrix

    def convolution(self, matrix):
        matrix = self.add_auto_padding(matrix)
        matrix = self.add_padding(matrix)

        height = len(matrix)
        width = len(matrix[0])

        conv = []
        self.init_biases()

        for z in range(self._filters):
            temp2 = []
            for i in range(0, height - self._kernel_size[0] + 1, self._strides[0]):
                temp1 = []
                for j in range(0, width - self._kernel_size[1] + 1, self._strides[1]):
                    sum = 0
                    for k in range(self._kernel_size[0]):
                        for l in range(self._kernel_size[1]):
                            sum += matrix[i + k][j + l] * self._weights[z][k][l]
                        sum += self._biases[z][k]
                    temp1.append(sum)
                temp2.append(temp1)
            conv.append(temp2)

        return conv

    def detector(self, matrix):
        return [self._activation(row).result for row in matrix]

    def forward_propagation(self, neurons):
        self.input_neurons = neurons
        convoluted = self.convolution(neurons[0])
        for i in range(1, len(neurons)):
            temp = self.convolution(neurons[i])
            for j in range(len(convoluted)):
                for k in range(len(convoluted[j])):
                    for l in range(len(convoluted[j][k])):
                        convoluted[j][k][l] += temp[j][k][l]

        detected = [self.detector(item) for item in convoluted]
        self.set_outputs_value_by_matrix(convoluted, detected)
    
