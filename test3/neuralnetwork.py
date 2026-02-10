from fileinput import filename
import numpy as np
import datasetloader as dl
import threading
import activation as act

class NeuralNetwork():
    def __init__(self, num_input: int, hidden_layers: dict[str, int], num_output: int):
        self.num_input = num_input
        self.hidden_layers = hidden_layers
        self.num_outputs = num_output
        self.weights = []
        self.biases = []
        self.forward: any = None
        self.dynamic_learning_rate: bool = False

        self.layer_names = list(self.hidden_layers.keys()) + ['output']
        self.layer_sizes = list(self.hidden_layers.values()) + [self.num_outputs]

        print(self.layer_names)
        print(self.layer_sizes)

    def get_network_configuration(self):
        return {
            "num_input": self.num_input,
            "hidden_layers": self.hidden_layers,
            "num_output": self.num_outputs
        }

    def initialize_adaptive_weights_and_biases(self):
        input_size = self.num_input
        for layer_name, num_neurons in zip(self.layer_names, self.layer_sizes):
            print(f"Creating layer: {layer_name} with {num_neurons} neurons... Input size: {input_size}")
            self.weights.append(np.random.uniform(-0.5, 0.5, (num_neurons, input_size)))
            self.biases.append(np.random.uniform(-0.5, 0.5, (num_neurons,)))
            input_size = num_neurons
    
    def implement_forward(self, x):
        activations = []

        for w, b, name in zip(self.weights, self.biases, self.layer_names):
            z = np.dot(x, w.T) + b 
            x = act.Operations.relu(z) if name != 'output' else act.Operations.softmax(z)
            activations.append(x)
        return activations
    
    def train(self, train_inputs, train_labels, learning_rate=0.001, epochs=20, dynamic_learning_rate=False, decay_epochs=4, batch_size=32):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dynamic_learning_rate = dynamic_learning_rate
        self.decay_epochs = decay_epochs
        self.batch_size = batch_size

        self.learn(learning_rate, epochs, dynamic_learning_rate, decay_epochs)

    def learn(self, learning_rate, epochs, dynamic_learning_rate, decay_epochs):
        initial_lr = learning_rate

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs} - Learning Rate: {learning_rate:.6f}")

            indices = np.arange(len(self.train_inputs))
            np.random.shuffle(indices)
            train_inputs = self.train_inputs[indices]
            train_labels = self.train_labels[indices]

            if dynamic_learning_rate and epoch % decay_epochs == 0 and epoch != 0:
                learning_rate *= 0.85
                decay_step = decay_epochs
                learning_rate = initial_lr * (learning_rate ** (epoch // decay_step))

            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for start in range(0, len(train_inputs), self.batch_size):
                end = start + self.batch_size

                x = train_inputs[start:end]        # (batch, 784)
                y_true = train_labels[start:end]   # (batch,)
                batch_len = len(x)
                total_samples += batch_len

                if start % (self.batch_size * 50) == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Step {start}/{len(train_inputs)}")

                # FORWARD
                activations = self.implement_forward(x)
                output_a = activations[-1]         # (batch, 62)

                # ACCURACY
                predicted = np.argmax(output_a, axis=1)
                correct_predictions += np.sum(predicted == y_true)

                # ONE HOT
                y_onehot = np.zeros((batch_len, self.num_outputs))
                y_onehot[np.arange(batch_len), y_true] = 1

                # LOSS
                loss = -np.sum(y_onehot * np.log(output_a + 1e-9)) / batch_len
                total_loss += loss

                # BACKPROP
                error = output_a - y_onehot   # (batch, 62)

                grad_weights = [None] * len(self.weights)
                grad_biases = [None] * len(self.biases)

                for layer_idx in reversed(range(len(self.weights))):
                    input_to_layer = x if layer_idx == 0 else activations[layer_idx - 1]
                    # input_to_layer: (batch, neurons_prev)

                    # MATRIX GRADIENT
                    grad_weights[layer_idx] = error.T @ input_to_layer / batch_len
                    grad_biases[layer_idx] = np.sum(error, axis=0) / batch_len

                    if layer_idx > 0:
                        error = error @ self.weights[layer_idx]  # (batch, neurons_prev)
                        error = error * (activations[layer_idx - 1] > 0).astype(float)

                # UPDATE
                for layer_idx in range(len(self.weights)):
                    self.weights[layer_idx] -= learning_rate * grad_weights[layer_idx]
                    self.biases[layer_idx] -= learning_rate * grad_biases[layer_idx]

            avg_loss = total_loss / (len(train_inputs) / self.batch_size)
            accuracy = correct_predictions / total_samples

            print(f"Epoch {epoch+1} done - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

        self.save()


    def save(self, filename="trained_model.npz"):
        save_dict = {}
        for i, w in enumerate(self.weights):
            save_dict[f'weight_{i}'] = w
        for i, b in enumerate(self.biases):
            save_dict[f'bias_{i}'] = b

        np.savez(filename, **save_dict)
    
    def predict(self, img: np.ndarray) -> tuple:
        flat_img = img.flatten()
        activations = self.implement_forward(flat_img)
        output = activations[-1]
        probs = act.Operations.softmax(output)
        #print("Softmax probabilities:", probs)
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        print(f"Predicted class: {predicted_class}, confidence: {confidence:.4f}")
        return predicted_class, confidence

    
    def load(self, filename="trained_model.npz"):
        data = np.load(filename)

        self.weights = [data[f'weight_{i}'] for i in range(len(self.layer_sizes))]
        self.biases = [data[f'bias_{i}'] for i in range(len(self.layer_sizes))]

        print(f"Model loaded from {filename}")

                