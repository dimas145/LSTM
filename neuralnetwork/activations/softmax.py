import math


class Softmax():
    def __init__(self, inputs):
        sum = 0
        inputs = list(map(lambda x: math.exp(x), inputs))

        for input in inputs:
            sum += input

        self._result = [input / sum for input in inputs]

    @property
    def result(self):
        return self._result
