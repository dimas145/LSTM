class Linear():
    def __init__(self, inputs):
        self._result = []

        for input in inputs:
            self._result.append(input)

    @property
    def result(self):
        return self._result
