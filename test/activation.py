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

    # Implement the forward pass for the neural network
    # This method takes weights, biases, and layer names as inputs
    # It returns a function that navigates the input image through the network
    # The function will apply the activation functions (ReLU, softmax, etc.) as specified
    @staticmethod
    def implement_forward(weights, biases, layer_names):
        print(weights, biases, layer_names)
        def forward(x):
            activations = []
            for i, (w, b, name) in enumerate(zip(weights, biases, layer_names)):
                z = np.dot(w, x) + b
                # print(f"Layer {name} - z min/max: {z.min()}, {z.max()}")
                x = Operations.relu(z) if name != 'output' else Operations.softmax(z)
                # print(f"Layer {name} - activation min/max: {x.min()}, {x.max()}")
                activations.append(x)
            return activations
        return forward

