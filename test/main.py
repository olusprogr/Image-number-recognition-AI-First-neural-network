import tensorflow as tf
import datasetloader as dl
import neuralnetwork as nn
import activation as act

train = True

if tf.config.list_physical_devices('GPU'):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if __name__  == '__main__':

    # Initialize the neural network with specified parameters
    print("Initializing neural network...")
    nn_model = nn.NeuralNetwork(
        num_inputs=784,
        hidden_layers={
            'hidden1': 128,
            'hidden2': 128
        },
        num_outputs=62
    )

    if train:
        # Load and preprocess the dataset
        # Normalize the images on a gray scale of 0-1
        print("Loading dataset...")
        dl_instance = dl.DatasetLoader(dataset_name='byclass')
        print("Normalizing dataset...")
        dl_instance.train_images = dl_instance.normalize_dataset(dl_instance.train_images) # (697932, 28, 28)
        dl_instance.test_images = dl_instance.normalize_dataset(dl_instance.test_images) # (116323, 28, 28)

        if (dl_instance.train_images is None or dl_instance.test_images is None):
            raise ValueError("Train or test images are None after normalization.")

        print(f"Train images shape: {dl_instance.train_images.shape}")  # (697932, 28, 28)
        print(f"Test images shape: {dl_instance.test_images.shape}")    # (116323, 28, 28)

        # Reshape the images to (num_samples, 784) for input to the neural network
        print("Reshaping images...")
        train_inputs = dl_instance.reshape_class_images(dl_instance.train_images)
        test_inputs = dl_instance.reshape_class_images(dl_instance.test_images)

        print(f"Train inputs shape: {train_inputs.shape}")  # (697932, 784)
        print(f"Test inputs shape: {test_inputs.shape}")    # (116323, 784)

        nn_model.initialize_adaptive_weights_and_biases()

        if (train_inputs[0].shape != (nn_model.num_inputs,)):
            raise ValueError(f"Input shape mismatch: expected ({nn_model.num_inputs},), got {train_inputs[0].shape}")

    else:
        nn_model.load(filename='trained_model.npz')

        print("Loaded model weights and biases:")
        print(nn_model.biases[-1].shape)   # Output layer biases shape


    activation_ops = act.Operations()

    if train:
        nn_model.train(
            train_inputs=train_inputs,
            train_labels=dl_instance.train_labels,
            learning_rate=0.003,
            epochs=1,
            dynamic_learning_rate=True,
            decay_epochs=5
        )

    else:
        print(nn_model.layer_names)
        print(nn_model.layer_sizes)

        imgs = ["Unbenannt.png"]
        results = []
        for img_path in imgs:
            img = dl.DatasetLoader.load_image(img_path)
            prediction = nn_model.predict(img=img)
            print(prediction)
            class_name = dl.DatasetLoader.number_to_emnist_class(prediction[0])
            results.append((img_path, prediction[0], class_name, prediction[1]))

        print("\nVorhersagen:")
        print(f"{'Bild':<20} {'Klasse':<8} {'Name':<15} {'Konfidenz':<10}")
        print("-" * 55)
        for img_path, pred_class, class_name, confidence in results:
            print(f"{img_path:<20} {pred_class:<8} {class_name:<15} {confidence:.4f}")

        # correct = 0
        # for x, y_true in zip(test_inputs, dl_instance.test_labels):
        #     pred, _ = nn_model.predict(x)
        #     if pred == y_true:
        #         correct += 1

        # accuracy = correct / len(test_inputs)
        # print(f"Test Accuracy: {accuracy:.2%}")
