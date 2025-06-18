import numpy as np

class Operations:

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def implement_forward(weights, biases, layer_names):
        def forward(x):
            activations = []
            for i, (w, b, name) in enumerate(zip(weights, biases, layer_names)):
                z = np.dot(w, x) + b
                x = Operations.relu(z) if name != 'output' else Operations.softmax(z)
                activations.append(x)
            return activations
        return forward

