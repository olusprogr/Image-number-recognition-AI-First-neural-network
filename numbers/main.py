import numpy as np
import tensorflow as tf
from emnist import list_datasets, extract_training_samples, extract_test_samples


list_dataset = list_datasets()
print(list_dataset)  # ['byclass', 'bymerge', 'letters', 'mnist', 'balanced', 'digits', 'letters', 'mnist_byclass', 'mnist_bymerge']


tf.debugging.set_log_device_placement(True)

print(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 1. Load and preprocess MNIST dataset
x_train, y_train = extract_training_samples(list_dataset[0])
x_test, y_test = extract_test_samples(list_dataset[0])

# Convert images to float32 and normalize to [0, 1]
def normalize_images(images):
    return images.astype(np.float32) / 255.0

# Reshape test amd train images to (num_samples, 28, 28, 1) for CNN input
train_images = normalize_images(x_train)
test_images = normalize_images(x_test)

train_inputs = train_images.reshape((train_images.shape[0], -1))  # (60000, 784)
test_inputs = test_images.reshape((test_images.shape[0], -1))     # (10000, 784)

# 2. Network parameter
num_inputs = 784
num_hidden = 128
num_outputs = 62

np.random.seed(42)  # idk

# Initialize weights and biases that can be configured during training
weights_input_hidden1 = np.random.uniform(-0.5, 0.5, (num_hidden, 784))
bias_hidden1 = np.random.uniform(-0.5, 0.5, (num_hidden,))

weights_hidden1_hidden2 = np.random.uniform(-0.5, 0.5, (num_hidden, num_hidden))
bias_hidden2 = np.random.uniform(-0.5, 0.5, (num_hidden,))

weights_hidden2_output = np.random.uniform(-0.5, 0.5, (num_outputs, num_hidden))
bias_output = np.random.uniform(-0.5, 0.5, (num_outputs,))

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
epochs = 10

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(train_inputs)):
        x = train_inputs[i]
        y_true = y_train[i]

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
    if pred == y_test[i]:
        correct += 1

print(f"Testgenauigkeit: {correct / len(test_inputs) * 100:.2f}%")

