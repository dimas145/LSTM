import numpy as np


class Flatten:
    def __init__(self, name="flatten"):
        self._name = name

        self._size = 0
        self._neurons = []
        self._input_shape = None
        self._output_shape = (None, self._size)

    def init_layer(self):
        self._output_shape = (None, self._input_shape[1] *
                              self._input_shape[2] * self._input_shape[3])
        self._size = int(self._input_shape[1] * self._input_shape[2] *
                         self._input_shape[3])

    @property
    def output_size(self):
        return self._size

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def neurons(self):
        return self._neurons

    @property
    def input_size(self):
        return self._input_shape

    @input_size.setter
    def input_size(self, shape):
        self._input_shape = shape
        self._size = int(self._input_shape[1] * self._input_shape[2] *
                         self._input_shape[3])

    def flattening(self, matrix):
        flattened = []

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                for k in range(len(matrix[i][j])):
                    flattened.append(matrix[i][j][k])

        self._neurons = flattened
