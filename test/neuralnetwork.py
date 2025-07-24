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

        # self.act = act.Operations()

    
    def initialize_adaptive_weights_and_biases(self):
        input_size = self.num_inputs
        for layer_name, num_neurons in zip(self.layer_names, self.layer_sizes):
            print(f"Creating layer: {layer_name} with {num_neurons} neurons")
            self.weights.append(np.random.uniform(-0.5, 0.5, (num_neurons, input_size)))
            self.biases.append(np.random.uniform(-0.5, 0.5, (num_neurons,)))
            input_size = num_neurons
        
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
            for i in range(len(train_inputs)):
                if i % 10000 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Step {i}/{len(train_inputs)} - Loss so far: {total_loss/(i+1):.4f}", flush=True)

                    current_avg_loss = total_loss / (i + 1)

                    if gui:
                        gui.root.after(0, lambda loss=current_avg_loss: gui.update_plot(loss))


                x = train_inputs[i]
                y_true = train_labels[i]

                activations = self.forward(x)

                output_a = activations[-1]  # Output Layer Activation

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

                

            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_inputs):.4f}")

            

    def save(self, filename="trained_model.npz"):
        save_dict = {}
        for i, w in enumerate(self.weights):
            save_dict[f'weight_{i}'] = w
        for i, b in enumerate(self.biases):
            save_dict[f'bias_{i}'] = b

        np.savez(filename, **save_dict)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, img: np.ndarray) -> tuple:
        activations = self.forward(img.flatten())
        output = activations[-1]
        probs = NeuralNetwork.softmax(output)
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        return predicted_class, confidence



