class ReLU():
    def __init__(self, inputs):
        self._result = []

        for input in inputs:
            self._result.append(max(input, 0))

    @property
    def result(self):
        return self._result
