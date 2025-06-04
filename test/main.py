import numpy as np
import tensorflow as tf
from emnist import extract_training_samples, extract_test_samples
from emnist import list_datasets

print(f"Available datasets: {list_datasets()}")

SELECTED_DATASET = 'byclass'

print(f"Selected dataset: {SELECTED_DATASET}")

if tf.config.list_physical_devices('GPU'):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if list_datasets() is None:
    raise RuntimeError("No datasets available. Please check your EMNIST installation.")

if SELECTED_DATASET not in list_datasets():
    raise ValueError(f"Dataset '{SELECTED_DATASET}' is not available. Available datasets: {list_datasets()}")

if __name__  != '__main__':
    raise RuntimeError("This script is not intended to be imported as a module.")

x_train, y_train = extract_training_samples(SELECTED_DATASET)
x_test, y_test = extract_test_samples(SELECTED_DATASET)

print(x_train.shape)  # (124800, 28, 28)
print(y_train.shape)  # (124800,)
print(x_test.shape)   # (20800, 28, 28)
print(y_test.shape)   # (20800,)

train_images, train_labels = x_train, y_train
test_images, test_labels = x_test, y_test

def normalize_images(images):
    return images.astype(np.float32) / 255.0

train_images = normalize_images(train_images)
test_images = normalize_images(test_images)

train_inputs = train_images.reshape((train_images.shape[0], -1))
test_inputs = test_images.reshape((test_images.shape[0], -1))

num_inputs = 784
hidden_layers = {
    'hidden1': 128,
    'hidden2': 128,
    'hidden3': 128
}
num_outputs = 26

weights = {}
biases = {}

layer_names = list(hidden_layers.keys()) + ['output']
layer_sizes = list(hidden_layers.values()) + [num_outputs]

input_size = num_inputs

for layer_name, num_neurons in zip(layer_names, layer_sizes):
    weights[layer_name] = np.random.uniform(-0.5, 0.5, (num_neurons, input_size))
    biases[layer_name] = np.random.uniform(-0.5, 0.5, (num_neurons,))
    input_size = num_neurons

for weights, biases in zip(weights.values(), biases.values()):
    print(f"Weights shape: {weights.shape}, Biases shape: {biases.shape}")

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

