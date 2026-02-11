import tensorflow as tf
import neuralnetwork as nn
import datasetloader as dl

train: bool = False

if tf.config.list_physical_devices('GPU'):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    


if __name__ == "__main__":
    nn_model = nn.NeuralNetwork(
        num_input=784,
        hidden_layers={
            'hidden1': 512,
            'hidden2': 256,
            'hidden3': 128,
            'hidden4': 64,
        },
        num_output=62
    )

    print(nn_model.get_network_configuration())

    if train:
        dl_instance = dl.DatasetLoader(dataset_name='byclass', normalize=True)

        if (dl_instance.train_images is None or dl_instance.test_images is None):
            raise ValueError("Train or test images are None after normalization.")
        
        print(f"Train images shape: {dl_instance.train_images.shape}")  # (697932, 28, 28)
        print(f"Test images shape: {dl_instance.test_images.shape}") # (116323, 28, 28)

        dl_instance.reshape_class_images()

        print(f"Train inputs shape: {dl_instance.train_inputs.shape}")  # (697932, 784)
        print(f"Test inputs shape: {dl_instance.test_inputs.shape}")    # (116323, 784)

        nn_model.initialize_adaptive_weights_and_biases()
    
    else:
        nn_model.load(filename='trained_model.npz')

    if train:
        nn_model.train(train_inputs=dl_instance.train_inputs, train_labels=dl_instance.train_labels, learning_rate=0.005, epochs=40, dynamic_learning_rate=True, decay_epochs=4, batch_size=128)
    
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
