import numpy as np

class Operations:

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        if x.ndim == 1:
            e_x = np.exp(x - np.max(x))
            return e_x / np.sum(e_x)
        else:
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / np.sum(e_x, axis=1, keepdims=True)



    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)