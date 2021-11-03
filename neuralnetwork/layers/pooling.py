import numpy as np
import copy

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

    def X_ReLU(conv, pool, pool_size, strides):
           
        _conv = copy.copy(conv)  

        for z in range(len(_conv)):
            for i in range(0, len(_conv[z]) - pool_size[0] + 1, strides[0]):
                for j in range(0, len(_conv[z][i]) - pool_size[1] + 1, strides[1]):
                    for k in range(pool_size[0]):
                        for l in range(pool_size[1]):
                            if(_conv[z][i + k][j + l] == pool[z][i//strides[0]][j//strides[1]]):
                                _conv[z][i + k][j + l] = 1
                            else:
                                _conv[z][i + k][j + l] = 0
        return np.array(_conv)

    def constant_mult_matrix(conv, pool, pool_size, strides):
        _conv = conv.tolist()
        count = 0
        
        for z in range(len(_conv)):
            for i in range(0, len(_conv[z]) - pool_size[0] + 1, strides[0]):
                for j in range(0, len(_conv[z][i]) - pool_size[1] + 1, strides[1]):
                    for k in range(pool_size[0]):
                        for l in range(pool_size[1]):
                            temp = _conv[z][i + k][j + l] * pool[count]
                            # print("=> " + str((z, i+k, j+l)) + " " + str(count) + " " + str(temp))
                            _conv[z][i + k][j + l] = temp
                    count += 1
        # print(np.array(_conv))
        return _conv

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


class Pooling:
    def __init__(
            self,
            pool_mode,
            name="pooling",
            pool_size=(2, 2),
            pool_strides=None,
            pool_padding=(0, 0),
    ):
        self._name = name

        self._pool_mode = pool_mode
        self._pool_strides = pool_strides if (pool_strides
                                             is not None) else pool_size
        self._pool_padding = pool_padding
        self._pool_size = pool_size

        self._neurons = []
        self._nets = []

        self._input_shape = None
        self._output_shape = None

        self._dE_do = None

    @property
    def output_size(self):
        return self._output_shape

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
    def nets(self):
        return self._nets

    @nets.setter
    def nets(self, nets):
        self._nets = nets

    @property
    def input_size(self):
        return self._input_shape
    
    @input_size.setter
    def input_size(self, shape):
        self._input_shape = shape

    def init_layer(self):
        height = self._input_shape[1]
        width = self._input_shape[2]

        if (width % self._pool_strides[1] != 0):
            width += self._pool_strides[1] - (width % self._pool_strides[1])

        if (height % self._pool_strides[0] != 0):
            height += self._pool_strides[0] - (height % self._pool_strides[0])

        self._output_shape = (None, width // self._pool_size[1],
                             height // self._pool_size[0], self._input_shape[3])

    def add_auto_padding(self, matrix):
        height = len(matrix)
        width = len(matrix[0])

        left_padding = 0
        right_padding = 0

        up_padding = 0
        down_padding = 0

        if (width % self._pool_strides[1] != 0):
            for i in range(self._pool_strides[1] -
                           (width % self._pool_strides[1])):
                if (i % 2 == 0):
                    right_padding += 1
                else:
                    left_padding += 1

        if (height % self._pool_strides[0] != 0):
            for i in range(self._pool_strides[0] -
                           (height % self._pool_strides[0])):
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
        width = len(matrix[0])

        left_padding = self._pool_padding[1]
        right_padding = self._pool_padding[1]

        up_padding = self._pool_padding[0]
        down_padding = self._pool_padding[0]

        for i in range(height):
            matrix[i] += [0] * right_padding
            for j in range(left_padding):
                matrix[i].insert(0, 0)

        for _ in range(up_padding):
            matrix.insert(0, [0] * len(matrix[0]))

        for _ in range(down_padding):
            matrix.append([0] * len(matrix[0]))

        return matrix

    def max_pooling(self, matrix):
        matrix = self.add_auto_padding(matrix)
        matrix = self.add_padding(matrix)

        height = len(matrix)
        width = len(matrix[0])

        pooled = []

        for i in range(0, height - self._pool_size[0] + 1,
                       self._pool_strides[0]):
            temp1 = []
            for j in range(0, width - self._pool_size[1] + 1,
                           self._pool_strides[1]):
                max = matrix[i][j]
                for k in range(self._pool_size[0]):
                    for l in range(self._pool_size[1]):
                        if (matrix[i + k][j + l] > max):
                            max = matrix[i + k][j + l]
                temp1.append(max)
            pooled.append(temp1)

        return pooled

    def average_pooling(self, matrix):

        matrix = self.add_auto_padding(matrix)
        matrix = self.add_padding(matrix)

        height = len(matrix)
        width = len(matrix[0])

        pooled = []

        for i in range(0, height - self._pool_size[0] + 1,
                       self._pool_strides[0]):
            temp1 = []
            for j in range(0, width - self._pool_size[1] + 1,
                           self._pool_strides[1]):
                sum = 0
                for k in range(self._pool_size[0]):
                    for l in range(self._pool_size[1]):
                        sum += matrix[i + k][j + l]
                temp1.append(sum / (self._pool_size[0] * self._pool_size[1]))
            pooled.append(temp1)

        return pooled

    def pooling(self, matrix):

        self.nets = matrix

        if (self._pool_mode == "max"):
            res = [self.max_pooling(matrix[i]) for i in range(len(matrix))]
        elif self._pool_mode == "average":
            res = [self.average_pooling(matrix[i]) for i in range(len(matrix))]
        else:
            raise Exception("Undefined pooling mode!")

        self._neurons = res
    