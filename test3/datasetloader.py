from emnist import list_datasets, extract_training_samples, extract_test_samples
import numpy as np
from PIL import Image

class DatasetLoader:
    def __init__(self, dataset_name='byclass', normalize=False):
        self.dataset_name = dataset_name
        self.available_datasets = list_datasets()
        print(self.available_datasets)

        if not self.available_datasets:
            raise RuntimeError("No datasets available. Please check your EMNIST installation.")

        self.load()
        if normalize:
            self.train_images = self.normalize_dataset(self.train_images)
            self.test_images = self.normalize_dataset(self.test_images)

    def load(self) -> None:
        if self.dataset_name not in self.available_datasets:
            raise ValueError(
                f"Dataset '{self.dataset_name}' is not available. "
                f"Available datasets: {self.available_datasets}"
            )

        x_train, y_train = extract_training_samples(self.dataset_name)
        x_test, y_test = extract_test_samples(self.dataset_name)

        self.train_images, self.train_labels = x_train, y_train
        self.test_images, self.test_labels = x_test, y_test

    def normalize_dataset(self, images):
        return images.astype(np.float32) / 255.0
    
    @staticmethod
    def normalize_images(images):
        return images.astype(np.float32) / 255.0

    def reshape_class_images(self):
        self.train_inputs = self.train_images.reshape((self.train_images.shape[0], -1))
        self.test_inputs = self.test_images.reshape((self.test_images.shape[0], -1))

    @staticmethod
    def reshape_images(images):
        return images.reshape((images.shape[0], -1))
    
    @staticmethod
    def load_image(filepath: str) -> np.ndarray:
        # 1. Bild öffnen & in Graustufen konvertieren
        img = Image.open(filepath).convert('L')

        # 2. Größe auf 28x28 ändern
        img = img.resize((28, 28))

        # 3. PIL-Bild zu NumPy-Array umwandeln
        img_array = np.array(img)

        # 4. Normalisieren
        img_array = DatasetLoader.normalize_images(img_array)

        # 5. Zu (1, 28, 28) machen → ein einzelnes Bild in Stapel einbetten
        img_array = np.expand_dims(img_array, axis=0)

        # 6. Reshape zu (1, 784)
        img_array = DatasetLoader.reshape_images(img_array)

        return img_array

    @staticmethod
    def number_to_emnist_class(num: int) -> str:
        if 0 <= num <= 9:
            return str(num)
        elif 10 <= num <= 35:
            return chr(ord('A') + (num - 10))
        elif 36 <= num <= 61:
            return chr(ord('a') + (num - 36))
        else:
            raise ValueError("Nummer außerhalb des gültigen Bereichs 0-61", num)
        
