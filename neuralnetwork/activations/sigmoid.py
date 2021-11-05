import math
import copy

class Sigmoid():
    def __init__(self, inputs):
        self._result = copy.copy(inputs)

        for i in range(len(self._result)):
            for j in range(len(self._result[i])):
                self._result[i][j] = 1 / (1 + math.exp(-1 * inputs[i][j]))


    @property
    def result(self):
        return self._result
