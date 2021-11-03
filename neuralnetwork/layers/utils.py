import numpy as np
import copy

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
