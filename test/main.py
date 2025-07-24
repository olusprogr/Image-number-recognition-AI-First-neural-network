import tensorflow as tf
import datasetloader as dl
import neuralnetwork as nn
import activation as act

train = True

if tf.config.list_physical_devices('GPU'):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if __name__  == '__main__':
    dl_instance = dl.DatasetLoader(dataset_name='byclass')
    dl_instance.train_images = dl_instance.normalize_dataset(dl_instance.train_images)
    dl_instance.test_images = dl_instance.normalize_dataset(dl_instance.test_images)

    print(f"Train images shape: {dl_instance.train_images.shape}")  # (124800, 28, 28)
    print(f"Test images shape: {dl_instance.test_images.shape}")    # (20800, 28, 28)

    train_inputs = dl_instance.reshape_class_images(dl_instance.train_images)
    test_inputs = dl_instance.reshape_class_images(dl_instance.test_images)

    print(f"Train inputs shape: {train_inputs.shape}")  # (124800, 784)
    print(f"Test inputs shape: {test_inputs.shape}")    # (20800, 784)

    nn_model = nn.NeuralNetwork(
        num_inputs=784,
        hidden_layers={
            'hidden1': 128,
            'hidden2': 64,
            'hidden3': 32
        },
        num_outputs=62
    )
    nn_model.initialize_adaptive_weights_and_biases()


    activation_ops = act.Operations()

    forward = activation_ops.implement_forward(
        weights=nn_model.weights,
        biases=nn_model.biases,
        layer_names=nn_model.layer_names
    )

    if (train_inputs[0].shape != (nn_model.num_inputs,)):
        raise ValueError(f"Input shape mismatch: expected ({nn_model.num_inputs},), got {train_inputs[0].shape}")

    if train:
        nn_model.train(
            train_inputs=train_inputs,
            train_labels=dl_instance.train_labels,
            learning_rate=0.003,
            epochs=30,
            dynamic_learning_rate=True,
            decay_epochs=5
        )

        nn_model.save(filename='trained_model.npz')
    
    else:
        imgs = ["Unbenannt.png"]
        results = []
        for img_path in imgs:
            img = dl_instance.load_image(img_path)
            prediction = nn_model.predict(img=img)
            class_name = dl_instance.number_to_emnist_class(prediction[0])
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
