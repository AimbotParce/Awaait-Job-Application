import os
import pickle
import time
from functools import partial
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity
from sklearn import svm
from sklearn.decomposition import PCA

from parseMNIST import loadMNIST


class PCASVM(svm.SVC):

    # Now, for the fun part (the actual solution):
    def fit(self, X, y, sample_weight=None):
        print("Starting SVM training...")

        # Fit the PCA to the data, this will be used to reduce the dimensionality of the data
        print(f"Current dataset has shape {X.shape}. Reducing dimensionality via PCA.")
        print("Fitting PCA with all components...")
        # For now we won't give it a num of components. We'll see how many components we need afterwards
        self.pca = PCA()
        self.pca.fit(X)
        print(r"Keeping principal components such that 80% of the variation is explained:")
        cumsum = np.cumsum(np.sort(self.pca.explained_variance_ratio_)[::-1])
        keep_components = np.where(cumsum > 0.8)[0][0]
        print(
            f"We need the first {keep_components} components to explain {cumsum[keep_components]*100:.2f}% of the variance"
        )

        print(f"Fitting PCA again with only {keep_components} components...")
        self.pca = PCA(n_components=keep_components)
        self.pca.fit(X)

        # Reduce the dimensionality of the data
        print("Reducing dimensionality...")
        reducedData = self.pca.transform(X)
        print("Done! New Shape:", reducedData.shape)

        # Finally fit the SVM to the reduced data
        print("Training SVM...")

        start = time.time()
        super().fit(reducedData, y, sample_weight)
        end = time.time()
        print(f"Done! Time: {end - start:.2f}s")

    def predict(self, X) -> np.ndarray:
        reducedData = self.pca.transform(X)
        return super().predict(reducedData)

    def decision_function(self, X) -> np.ndarray:
        reducedData = self.pca.transform(X)
        return super().decision_function(reducedData)

    def save(self, path: os.PathLike):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path: os.PathLike):
        with open(path, "rb") as f:
            return pickle.load(f)


class Dataset:
    def __init__(self, mnist_path: os.PathLike):
        self.mnist_path = mnist_path
        self.train_images, self.train_labels, self.test_images, self.test_labels = loadMNIST(self.mnist_path)

        self._train_data, self._test_data = None, None

    @property
    def train_data(self):
        if self._train_data is None:
            self._train_data = self.train_images.reshape(len(self.train_images), -1)
        return self._train_data

    @property
    def test_data(self):
        if self._test_data is None:
            self._test_data = self.test_images.reshape(len(self.test_images), -1)
        return self._test_data


class App:
    def __init__(self, mnist_path: os.PathLike, model_path: os.PathLike = None):
        self.dataset = Dataset(mnist_path=mnist_path)
        if model_path is None:
            self.svm = PCASVM(kernel="rbf")
            self.is_svm_trained = False
        else:
            self.svm = PCASVM.load(model_path)
            self.is_svm_trained = True

        self._test_image_predictions = None
        self._test_accuracy = None

    def fitSVM(self):
        self.svm.fit(self.dataset.train_data, self.dataset.train_labels)
        self.is_svm_trained = True
        self._test_image_predictions = None  # If it was retrained, delete previous cached predictions

    @property
    def test_image_predictions(self):
        if self._test_image_predictions is None:
            self._test_image_predictions = self._predictTestImages()
        return self._test_image_predictions

    def _findSimilarSSIM(self, img: np.ndarray) -> np.ndarray:
        """
        Look inside the MNIST dataset for 10 cases representing the same number as the image img.
        This function uses a structural similarity measure to find the most similar images.
        This is of course not a perfect solution, but it's by far the simplest to implement.

        @param img: The image to be compared as a 28*28*1 numpy array.

        @return: A 10x28x28 array containing the 10 similar images.
        """
        # Most likely this wasn't the intended way to solve the problem, but it's clearly a simple solution.

        # Compute structural similarity
        ssimFunc = partial(structural_similarity, im2=img, multichannel=False)
        ssim = np.array(list(map(ssimFunc, self.dataset.train_images)))
        # Find the 10 most similar images
        indices = np.argsort(ssim)[-10:]
        return self.dataset.train_images[indices]

    def _findSimilarSVM(self, img: np.ndarray) -> np.ndarray:
        """
        Look inside the MNIST test dataset for 10 cases representing the same number as the image img.
        Labels of the test images are not used by any means in this function.
        This function uses the SVM to make predictions on the images.

        @param img: The image to be compared as a 28*28*1 numpy array.

        @return: A 10x28x28 array containing the 10 similar images.
        """
        # Predict the number
        predictedNumber = self.predict(img[np.newaxis, ...])

        # Find images with the same number
        test_numbers, test_confidences = self.test_image_predictions
        indices = np.where(test_numbers == predictedNumber)[0]
        imgs = self.dataset.test_images[indices]
        confidences = test_confidences[indices]

        # Sort the images by confidence
        sortedIndices = np.argsort(confidences)[-10:]
        return imgs[sortedIndices]

    def findSimilar(self, img: np.ndarray, strategy: Literal["ssim", "svm"] = "svm") -> np.ndarray:
        if strategy == "svm":
            return self._findSimilarSVM(img)
        elif strategy == "ssim":
            return self._findSimilarSSIM(img)
        else:
            raise ValueError(f"Unknown strategy {strategy}")

    def _predictTestImages(self):
        """
        Predict the number of the test images.
        """
        results = self.predict(self.dataset.test_data)
        confidences = self.confidence(self.dataset.test_data)
        return results, confidences

    @property
    def test_accuracy(self):
        if self._test_accuracy is None:
            self._test_accuracy = self._computeTestAccuracy()
        return self._test_accuracy

    def _computeTestAccuracy(self) -> float:
        """
        Test the accuracy of the SVM by predicting the number of the test images.

        @return: The accuracy of the SVM.
        """
        # Predict test images
        test_numbers, test_confidences = self.test_image_predictions

        # Compute the accuracy
        correct = test_numbers == self.dataset.test_labels
        accuracy = np.sum(correct) / len(correct)
        return accuracy

    def predict(self, X: np.ndarray):
        if not self.is_svm_trained:
            raise ValueError("SVM is not yet trained")
        reshaped = X.reshape(len(X), -1)
        return self.svm.predict(reshaped)

    def confidence(self, X: np.ndarray):
        if not self.is_svm_trained:
            raise ValueError("SVM is not yet trained")
        reshaped = X.reshape(len(X), -1)
        return np.max(self.svm.decision_function(reshaped), axis=1)


def main_ssim():
    """
    Run some tests on finding similar images using the structural similarity measure.
    """

    app = App(mnist_path="./uncompressed_data")

    img = app.dataset.test_images[3450]  # Arbitrary image from the set
    similar = app.findSimilar(img, strategy="ssim")

    fig, ax = plt.subplots(2, 6, figsize=(10, 4))
    ax[0, 0].imshow(img, cmap="gray")
    ax[0, 0].set_title(f"Original image")
    for i in range(5):
        ax[0, i + 1].imshow(similar[i], cmap="gray")
    for i in range(5):
        ax[1, i + 1].imshow(similar[i + 5], cmap="gray")
    plt.show()


def main_svm():
    """
    Actual solution: Run some tests on finding similar images using the SVM.
    """

    # Reshape train images to be a 2D array
    if os.path.exists("./trained_model.pkl"):
        app = App(mnist_path="./uncompressed_data", model_path="./trained_model.pkl")
        print("Loaded SVM")
    else:
        app = App(mnist_path="./uncompressed_data")

    if not app.is_svm_trained:
        app.fitSVM()
        print("Saving trained SVM")
        app.svm.save("./trained_model.pkl")

    print("Computing accuracy...")
    # Just for fun, will check the accuracy of the program.
    start = time.time()
    accuracy = app.test_accuracy
    end = time.time()
    print(f"Done! Accuracy: {accuracy*100:.2f}%, Time: {end - start:.2f}s")

    img = app.dataset.test_images[3]  # Arbitrary image from the set
    prediction = app.predict(img[np.newaxis, ...])[0]
    confidence = app.confidence(img[np.newaxis, ...])[0]

    print(prediction, confidence)
    print(f"Prediction: {prediction}, confidence: {confidence:.2f}")

    similar = app.findSimilar(img, strategy="svm")

    fig, ax = plt.subplots(2, 6, figsize=(10, 4))
    ax[0, 0].imshow(img, cmap="gray")
    ax[0, 0].set_title(f"Original (pred={prediction})")
    for i in range(5):
        ax[0, i + 1].imshow(similar[i], cmap="gray")
    for i in range(5):
        ax[1, i + 1].imshow(similar[i + 5], cmap="gray")
    plt.show()

    # I have written a simple and fun example on a notebook to test the SVM.


if __name__ == "__main__":
    main_svm()
