import sys
import math
import numpy as np
import pickle

from .layers import (
    Dense,
    Conv2D,
    Flatten,
    Pooling,
    LSTM
)

class Sequential:
    def __init__(self, layers=None):
        self.layers = []
        if layers != None:
            for layer in layers:
                self.add(layer)

        self.state = {Dense: 0, Conv2D: 0, Flatten: 0, Pooling: 0, LSTM: 0}
        self.loss = sys.maxsize

    def save_model(self, path):
        pickle.dump(self, open(path, "wb"))
        print("Model saved successfully!")

    def load_model(self, path):
        res = pickle.load(open(path, "rb"))
        print("Model loaded successfully!")
        return res

    def add(self, layer):
        if (len(self.layers) != 0):
            layer.input_size = self.layers[-1].output_size

        self.state[type(layer)] += 1

        if type(layer) in [Dense, Conv2D, Flatten, Pooling, LSTM]:
            layer.name += "_" + str(self.state[type(layer)])

        layer.init_layer()

        self.layers.append(layer)

    def forward_propagation(self, X, y=0):

        for k in range(len(self.layers)):
            if (type(self.layers[k]) == Dense):
                if (k == 0):
                    self.layers[k].forward_propagation([0] + X)
                else:
                    self.layers[k].forward_propagation([0] + self.layers[k - 1].neurons)
            elif (type(self.layers[k]) == Conv2D):
                if (k == 0):
                    self.layers[k].forward_propagation(X)
                else:
                    self.layers[k].forward_propagation(self.layers[k - 1].neurons)
            elif (type(self.layers[k]) == LSTM):
                if (k == 0):
                    self.layers[k].forward_propagation(X)
                else:
                    self.layers[k].forward_propagation(self.layers[k - 1].neurons)
            elif (type(self.layers[k]) == Flatten):
                self.layers[k].flattening(self.layers[k - 1].neurons)
            elif (type(self.layers[k]) == Pooling):
                self.layers[k].pooling(self.layers[k - 1].neurons)
                
        self.loss = -math.log(self.layers[-1].neurons[y])

    def predict(self, X):
        for i in range(X.shape[0]):
            res = []
            for k in range(len(self.layers)):
                if (type(self.layers[k]) == Dense):
                    if (k == 0):
                        self.layers[k].forward_propagation([0] + X[i])
                    else:
                        self.layers[k].forward_propagation([0] + self.layers[k - 1].neurons)
                elif (type(self.layers[k]) == Conv2D):
                    if (k == 0):
                        self.layers[k].forward_propagation(X[i])
                    else:
                        self.layers[k].forward_propagation(self.layers[k - 1].neurons)
                elif (type(self.layers[k]) == LSTM):
                    if (k == 0):
                        self.layers[k].forward_propagation(X[i])
                    else:
                        self.layers[k].forward_propagation(self.layers[k - 1].neurons)
                elif (type(self.layers[k]) == Flatten):
                    self.layers[k].flattening(self.layers[k - 1].neurons)
                elif (type(self.layers[k]) == Pooling):
                    self.layers[k].pooling(self.layers[k - 1].neurons)
            res.append(self.layers[-1].neurons)
        return np.array(res)
        # for i in range(len(self.layers[-1]._neurons)):
        #     if(self.layers[-1]._neurons[i] > maxi):
        #         maxi = self.layers[-1]._neurons[i]
        #         maxidx = i
                
        # return maxidx

    def summary(self):
        col1 = 35
        col2 = 35
        col3 = 17

        print()

        print("Model: Sequential")
        print("=" * 80)
        print("Layer (type)" + " " * 23 + "Output Shape" + " " * 23 +
              "Param #" + " " * 7)
        print("=" * 80)

        total_params = 0

        for i in range(len(self.layers)):
            layer = self.layers[i]

            if (type(layer) == Dense):
                before = self.layers[i].input_size
                param = (before + 1) * self.layers[i].output_size
            elif (type(layer) == LSTM):
                n = self.layers[i]._units
                m = self.layers[i]._features
                param = (m+n+1)*4*n
            elif (type(layer) == Conv2D):
                param = self.layers[i].output_size[3] * (
                    self.layers[i].kernel_size[0] *
                    self.layers[i].kernel_size[1] *
                    self.layers[i].input_shape[3] + 1)
            else:
                param = 0

            col1_text = self.layers[i].name + " " + "(" + type(
                self.layers[i]).__name__ + ")"
            if (type(layer) == Dense or type(layer) == Flatten or type(layer) == LSTM):
                col2_text = str((None, self.layers[i].output_size))
            else:
                col2_text = str(self.layers[i].output_size)
            col3_text = str(param)

            print(col1_text + " " * (col1 - len(col1_text)) + col2_text + " " *
                  (col2 - len(col2_text)) + col3_text + " " *
                  (col3 - len(col3_text)))

            if (i != len(self.layers) - 1):
                print("-" * 80)
            else:
                print("=" * 80)

            total_params += param

        print("Total params: " + "{:,}".format(total_params))
        print()
    
    def weights_summary(self):
        for i in range(len(self.layers)):
            if(type(self.layers[i]) == Dense or type(self.layers[i]) == Conv2D):
                w = np.array(self.layers[i]._weights)
                print(self.layers[i]._name + " " + str(w.shape))
                print(w)
            
        
