import tensorflow as tf

import datasetloader as dl
import neuralnetwork as nn
import activation as act

if tf.config.list_physical_devices('GPU'):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if __name__  == '__main__':
    dl = dl.DatasetLoader(dataset_name='byclass')
    dl.train_images = dl.normalize_dataset(dl.train_images)
    dl.test_images = dl.normalize_dataset(dl.test_images)

    train_inputs = dl.reshape_class_images(dl.train_images)
    test_inputs = dl.reshape_class_images(dl.test_images)

    nn = nn.NeuralNetwork(
        num_inputs=784,
        hidden_layers={
            'hidden1': 64,
            'hidden2': 64,
            'hidden3': 64
        },
        num_outputs=62
    )
    nn.initialize_adaptive_weights_and_biases()

    act = act.Operations()

    forward = act.implement_forward(
        weights=nn.weights,
        biases=nn.biases,
        layer_names=nn.layer_names
    )
    
    if (train_inputs[0].shape != (nn.num_inputs,)):
        raise ValueError(f"Input shape mismatch: expected ({nn.num_inputs},), got {train_inputs[0].shape}")


    nn.learn(
        train_inputs=train_inputs,
        train_labels=dl.train_labels,
        learning_rate=0.001,
        epochs=10
    )

    nn.save(filename='trained_model.npz')
