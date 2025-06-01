import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist



# 1. Daten laden und vorbereiten
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def normalize_images(images):
    return images.astype(np.float32) / 255.0

train_images = normalize_images(train_images)
test_images = normalize_images(test_images)

train_inputs = train_images.reshape((train_images.shape[0], -1))  # (60000, 784)
test_inputs = test_images.reshape((test_images.shape[0], -1))     # (10000, 784)

# 2. Netzwerk-Parameter
num_inputs = 784
num_hidden = 64
num_outputs = 10

np.random.seed(42)  # für Reproduzierbarkeit

weights_input_hidden1 = np.random.uniform(-0.5, 0.5, (64, 784))
bias_hidden1 = np.random.uniform(-0.5, 0.5, (64,))

weights_hidden1_hidden2 = np.random.uniform(-0.5, 0.5, (64, 64))
bias_hidden2 = np.random.uniform(-0.5, 0.5, (64,))

weights_hidden2_output = np.random.uniform(-0.5, 0.5, (10, 64))
bias_output = np.random.uniform(-0.5, 0.5, (10,))


# 3. Aktivierungsfunktionen
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# 4. Vorwärtsdurchlauf
def forward(x):
    hidden1_z = np.dot(weights_input_hidden1, x) + bias_hidden1
    hidden1_a = relu(hidden1_z)

    hidden2_z = np.dot(weights_hidden1_hidden2, hidden1_a) + bias_hidden2
    hidden2_a = relu(hidden2_z)

    output_z = np.dot(weights_hidden2_output, hidden2_a) + bias_output
    output_a = softmax(output_z)

    return hidden1_a, hidden2_a, output_a

# 5. Training
learning_rate = 0.005
epochs = 20

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(train_inputs)):
        x = train_inputs[i]
        y_true = train_labels[i]

        # Forward Pass - hier alle 3 Outputs auspacken!
        hidden1_a, hidden2_a, output_a = forward(x)

        # One-Hot Label
        y_onehot = np.zeros(num_outputs)
        y_onehot[y_true] = 1

        # Verlust (Cross-Entropy)
        loss = -np.sum(y_onehot * np.log(output_a + 1e-9))
        total_loss += loss

        error_output = output_a - y_onehot  # Fehler im Output

        grad_w_hidden2_output = np.outer(error_output, hidden2_a)
        grad_b_output = error_output

        error_hidden2 = np.dot(weights_hidden2_output.T, error_output)
        error_hidden2[hidden2_a <= 0] = 0  # ReLU Ableitung

        grad_w_hidden1_hidden2 = np.outer(error_hidden2, hidden1_a)
        grad_b_hidden2 = error_hidden2

        error_hidden1 = np.dot(weights_hidden1_hidden2.T, error_hidden2)
        error_hidden1[hidden1_a <= 0] = 0  # ReLU Ableitung

        grad_w_input_hidden1 = np.outer(error_hidden1, x)
        grad_b_hidden1 = error_hidden1

        # Update Gewichte und Biases
        weights_hidden2_output -= learning_rate * grad_w_hidden2_output
        bias_output -= learning_rate * grad_b_output

        weights_hidden1_hidden2 -= learning_rate * grad_w_hidden1_hidden2
        bias_hidden2 -= learning_rate * grad_b_hidden2

        weights_input_hidden1 -= learning_rate * grad_w_input_hidden1
        bias_hidden1 -= learning_rate * grad_b_hidden1

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_inputs):.4f}")

# Modell speichern (mit den korrekten Variablennamen)
np.savez("trained_model.npz",
         weights_input_hidden1=weights_input_hidden1,
         bias_hidden1=bias_hidden1,
         weights_hidden1_hidden2=weights_hidden1_hidden2,
         bias_hidden2=bias_hidden2,
         weights_hidden2_output=weights_hidden2_output,
         bias_output=bias_output)

# 6. Testen
correct = 0
for i in range(len(test_inputs)):
    _, _, output_a = forward(test_inputs[i])  # Drei Werte zurückgeben
    pred = np.argmax(output_a)
    if pred == test_labels[i]:
        correct += 1

print(f"Testgenauigkeit: {correct / len(test_inputs) * 100:.2f}%")

