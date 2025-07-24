from emnist import list_datasets, extract_training_samples, extract_test_samples
import numpy as np
from PIL import Image


class DatasetLoader:
    def __init__(self, dataset_name='byclass', available_datasets=None):
        self.dataset_name = dataset_name
        self.list_datasets = available_datasets or list_datasets()
        self.load()

    def load(self):
        print(type(self.list_datasets))
        print(self.list_datasets)
        print(self.list_datasets[0])

        if self.list_datasets is None:
             raise RuntimeError("No datasets available. Please check your EMNIST installation.")

        if self.dataset_name not in self.list_datasets:
             raise ValueError(f"Dataset '{self.dataset_name}' is not available. Available datasets: {list_datasets()}")
        
        x_train, y_train = extract_training_samples(self.dataset_name)
        x_test, y_test = extract_test_samples(self.dataset_name)

        self.train_images, self.train_labels = x_train, y_train
        self.test_images, self.test_labels = x_test, y_test
    
    def get_train_data(self) -> tuple:
        return self.train_images, self.train_labels
    
    def get_test_data(self) -> tuple:
        return self.test_images, self.test_labels
    
    def set_test_data(self, images, labels) -> None:
        self.test_images = images
        self.test_labels = labels

    def set_train_data(self, images, labels) -> None:
        self.train_images = images
        self.train_labels = labels


    def normalize_dataset(self, images):
        return images.astype(np.float32) / 255.0

    @staticmethod
    def normalize_images(images):
        return images.astype(np.float32) / 255.0
    
    def reshape_class_images(self, images):
        return images.reshape((images.shape[0], -1))
    
    @staticmethod
    def reshape_images(images):
        return images.reshape((images.shape[0], -1))
    
    @staticmethod
    def load_image(filepath: str) -> np.ndarray:
        img = Image.open(filepath).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img).astype(np.float32) / 255.0

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
            raise ValueError("Nummer außerhalb des gültigen Bereichs 0-61")



