import numpy as np
import activation as act
import datasetloader as dl
from loss_gui import LossGUI
import threading


class NeuralNetwork:
    def __init__(self, num_inputs:int, hidden_layers:dict, num_outputs:int):
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.weights = []
        self.biases = []
        self.forward = None
        self.dynamic_learning_rate = False

        
        self.layer_names = list(self.hidden_layers.keys()) + ['output']
        self.layer_sizes = list(self.hidden_layers.values()) + [self.num_outputs]

        print(self.layer_names)
        print(self.layer_sizes)




    # Initialize weights and biases
    # This method creates the weights and biases for each layer based on the specified sizes
    # The weights are initialized with random values in the range [-0.5, 0.5]
    # The biases are also initialized with random values in the same range
    def initialize_adaptive_weights_and_biases(self):
        input_size = self.num_inputs
        for layer_name, num_neurons in zip(self.layer_names, self.layer_sizes):
            print(f"Creating layer: {layer_name} with {num_neurons} neurons... Input size: {input_size}")
            self.weights.append(np.random.uniform(-0.5, 0.5, (num_neurons, input_size)))
            self.biases.append(np.random.uniform(-0.5, 0.5, (num_neurons,)))
            input_size = num_neurons

        # Implement the forward function using the activation operations
        # This allows the forward function to use the correct weights and biases for each layer
        print("Forward function is being implemented...")
        self.forward = act.Operations().implement_forward(
            weights=self.weights,
            biases=self.biases,
            layer_names=self.layer_names
        )
    
    def getNetworkSettings(self):
        return {
            'num_inputs': self.num_inputs,
            'hidden_layers': self.hidden_layers,
            'num_outputs': self.num_outputs,
            'weights': self.weights,
            'biases': self.biases
        }
    

    def train(self, train_inputs, train_labels, learning_rate=0.001, epochs=20, dynamic_learning_rate=False, decay_epochs=4):
        gui = LossGUI()

        def training_loop():
            self.learn(train_inputs, train_labels, learning_rate, epochs, dynamic_learning_rate, decay_epochs, gui)

        threading.Thread(target=training_loop, daemon=True).start()
        gui.start()


    def learn(self, train_inputs, train_labels, learning_rate=0.001, epochs=20, dynamic_learning_rate=False, decay_epochs=4, gui=None):
        print(f"Training with {len(train_inputs)} samples for {epochs} epochs at learning rate {learning_rate}")

        avg_loss = 0

        if gui:
            gui.root.after(0, lambda loss=avg_loss: gui.update_plot(loss))

        initial_lr = learning_rate
        self.dynamic_learning_rate = dynamic_learning_rate

        for epoch in range(epochs):

            indices = np.arange(len(train_inputs))
            np.random.shuffle(indices)
            train_inputs = train_inputs[indices]
            train_labels = train_labels[indices]


            if dynamic_learning_rate:
                decay_rate = 0.85
                decay_step = decay_epochs
                learning_rate = initial_lr * (decay_rate ** (epoch // decay_step))

            total_loss = 0
            correct_predictions = 0

            for i in range(len(train_inputs)):
                if i % 10000 == 0 and i > 0:
                    print(f"Epoch {epoch+1}/{epochs} - Step {i}/{len(train_inputs)} - Loss so far: {total_loss/(i+1):.4f}", flush=True)

                    current_avg_loss = total_loss / (i + 1)

                    accuracy_so_far = correct_predictions / (i + 1)

                    print(accuracy_so_far)

                    if gui:
                        gui.root.after(0, lambda loss=current_avg_loss, acc=accuracy_so_far: gui.update_plot(loss, cross_entropy=acc))


                x = train_inputs[i]
                y_true = train_labels[i]

                activations = self.forward(x)

                output_a = activations[-1]  # Output Layer Activation

                predicted_class = np.argmax(output_a)
                if predicted_class == y_true:
                    correct_predictions += 1

                y_onehot = np.zeros(self.num_outputs)
                y_onehot[y_true] = 1

                # Verlust (Cross-Entropy)
                loss = -np.sum(y_onehot * np.log(output_a + 1e-9))
                total_loss += loss

                error_output = output_a - y_onehot  # Fehler im Output
                error = error_output

                grad_weights = [None] * len(self.weights)
                grad_biases = [None] * len(self.biases)

                for layer_idx in reversed(range(len(self.weights))):
                    # Aktivierung der vorherigen Schicht (Input für aktuelle Gewichte)
                    input_to_layer = x if layer_idx == 0 else activations[layer_idx - 1]

                    # Gradienten für Gewicht und Bias berechnen
                    grad_weights[layer_idx] = np.outer(error, input_to_layer)
                    grad_biases[layer_idx] = error

                    if layer_idx > 0:
                        # Fehler für vorherige Schicht berechnen
                        error = np.dot(self.weights[layer_idx].T, error)

                        # ReLU Ableitung: Fehler nur weitergeben, wo Aktivierung > 0 war
                        error = error * (activations[layer_idx - 1] > 0).astype(float)

                for layer_idx in range(len(self.weights)):
                    self.weights[layer_idx] -= learning_rate * grad_weights[layer_idx]
                    self.biases[layer_idx] -= learning_rate * grad_biases[layer_idx]

            avg_loss = total_loss / len(train_inputs)
            accuracy = correct_predictions / len(train_inputs)


            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_inputs):.4f}")

        print("Training complete.")
        self.save(filename='trained_model.npz')

        if gui:
            gui.close()

            

    def save(self, filename="trained_model.npz"):
        save_dict = {}
        for i, w in enumerate(self.weights):
            save_dict[f'weight_{i}'] = w
        for i, b in enumerate(self.biases):
            save_dict[f'bias_{i}'] = b

        np.savez(filename, **save_dict)

    def predict(self, img: np.ndarray) -> tuple:
        flat_img = img.flatten()
        print("Input vector min/max:", flat_img.min(), flat_img.max())
        activations = self.forward(flat_img)
        output = activations[-1]
        probs = act.Operations.softmax(output)
        print("Softmax probabilities:", probs)
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        print(f"Predicted class: {predicted_class}, confidence: {confidence:.4f}")
        return predicted_class, confidence


    def load(self, filename="trained_model.npz"):
        data = np.load(filename)
        self.weights = [data[f'weight_{i}'] for i in range(len(self.layer_sizes) - 1)]
        self.biases = [data[f'bias_{i}'] for i in range(len(self.layer_sizes) - 1)]
        
        # Re-implement the forward function with the loaded weights and biases
        self.forward = act.Operations().implement_forward(
            weights=self.weights,
            biases=self.biases,
            layer_names=self.layer_names
        )
        print(f"Model loaded from {filename}")



