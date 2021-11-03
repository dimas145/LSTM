import math


class Sigmoid():
    def __init__(self, inputs):
        self._result = []

        for input in inputs:
            s = 1 / (1 + math.exp(-1 * input))

            self._result.append(s)

    @property
    def result(self):
        return self._result
