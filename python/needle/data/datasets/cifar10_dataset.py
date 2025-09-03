import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.base_folder = base_folder
        self.train = train

        # Load the data
        if self.train:
            self.X, self.y = self._load_train_data()
        else:
            self.X, self.y = self._load_test_data()

        # Normalize the images to 0-1 range
        self.X = self.X / 255.0

    def _load_batch(self, file_name: str):
        """Loads a single batch of data."""
        with open(file_name, 'rb') as file:
            batch = pickle.load(file, encoding='latin1')
            images = batch['data'].reshape(-1, 3, 32, 32)
            labels = batch['labels']
        return images, labels

    def _load_train_data(self):
        """Loads all the training data batches."""
        X_list = []
        y_list = []
        for i in range(1, 6):
            file_name = os.path.join(self.base_folder, f'data_batch_{i}')
            images, labels = self._load_batch(file_name)
            X_list.append(images)
            y_list.append(labels)
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        return X, y

    def _load_test_data(self):
        """Loads the test data."""
        file_name = os.path.join(self.base_folder, 'test_batch')
        X, y = self._load_batch(file_name)
        return X, y
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        X_items = self.X[index]
        Y_items = self.y[index]
        if isinstance(index, (slice, np.ndarray)):
            Y_items = np.reshape(Y_items, (Y_items.shape[0]))
            X_items = np.reshape(X_items, (X_items.shape[0], 3, 32, 32))
            for item_idx in range(X_items.shape[0]):
                X_item = X_items[item_idx]
                X_items[item_idx] = self.apply_transforms(X_item)
        else:
            X_items = np.reshape(X_items, (3, 32, 32))
            X_items = self.apply_transforms(X_items)
        return (X_items, Y_items)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION
