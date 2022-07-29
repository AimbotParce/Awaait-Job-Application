import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity
from sklearn import svm
from sklearn.decomposition import PCA

from parseMNIST import loadMNIST


class App:
    def __init__(self, mnist_path:str = "."):
        self.mnistPath = mnist_path
        self.loadData()

        # Initialize variable to store the predictions, this will be computed only once
        self.testPredNumbers = None

    def loadData(self):
        self.trainImages, self.trainLabels, self.testImages, self.testLabels = loadMNIST(self.mnistPath)

    def findSimilar_ssim(self, img: np.ndarray) -> np.ndarray:
        """
        Look inside the MNIST dataset for 10 cases representing the same number as the image img.
        This function uses a structural similarity measure to find the most similar images.
        This is of course not a perfect solution, but it's by far the simplest to implement.

        @param img: The image to be compared as a 28*28*1 numpy array.

        @return: A 10x28x28 array containing the 10 similar images.
        """
        # Most likely this wasn't the intended way to solve the problem, but it's clearly a simple solution.

        # Compute structural similarity
        ssim = [structural_similarity(img, self.trainImages[i], multichannel=False) for i in range(len(self.trainImages))]

        # Find the 10 most similar images
        indices = np.argsort(ssim)[-10:]
        return self.trainImages[indices]

    # Now, for the fun part (the actual solution):
    def trainSVM(self):
        """
        Train a SVM to make predictions on the image of a number.
        """
        print("Training SVM...")
        # Create a SVM
        self.svm = svm.SVC(kernel='rbf')

        # Reshape train images to be a 2D array
        self.trainData = self.trainImages.reshape(len(self.trainImages), -1)

        # Fit the PCA to the data, this will be used to reduce the dimensionality of the data
        print("Fitting PCA...")
        self.pca = PCA(n_components=.9)
        self.pca.fit(self.trainData)

        # Reduce the dimensionality of the data
        print("Reducing dimensionality...")
        reducedData = self.pca.transform(self.trainData)
        print("Done! New Shape:", reducedData.shape)

        # Finally fit the SVM to the reduced data
        print("Training SVM...")

        start = time.time()
        self.svm.fit(reducedData, self.trainLabels)
        end = time.time()
        print(f"Done! Time: {end - start:.2f}s")

    def predict(self, img: np.ndarray) -> Tuple[int, float]:
        """
        Predict the number of an image img and compute the confidence.
        Note that the confidence is not a percentage, but a distance to the decision boundary.

        @param img: The image to be predicted as a 28*28*1 numpy array.

        @return: A tuple containing the predicted number and the confidence of the prediction.
        """
        # Reshape the image to be a 2D array
        img = img.reshape(1, -1)

        # Reduce the dimensionality of the image
        reducedImg = self.pca.transform(img)

        # Predict the number
        prediction = self.svm.predict(reducedImg)

        # Compute confidence
        confidence = self.svm.decision_function(reducedImg)

        return prediction, np.max(confidence)

    def predict_test_images(self):
        """
        Predict the number of the test images.
        """

        # There's no need to compute again the predictions, to save time
        if self.testPredNumbers is not None:
            return 

        print("Predicting test images...")

        # Reshape test images
        testData = self.testImages.reshape(len(self.testImages), -1)
        # Transform with pca
        reducedData = self.pca.transform(testData)
        
        # Save predictions and confidences
        self.testPredNumbers = self.svm.predict(reducedData)
        confidences = self.svm.decision_function(reducedData)
        self.testConfidences = np.max(confidences, axis=1)

    def findSimilar_svm(self, img: np.ndarray) -> np.ndarray:
        """
        Look inside the MNIST test dataset for 10 cases representing the same number as the image img.
        Labels of the test images are not used by any means in this function.
        This function uses the SVM to make predictions on the images.

        @param img: The image to be compared as a 28*28*1 numpy array.

        @return: A 10x28x28 array containing the 10 similar images.
        """
        # Predict the number
        predictedNumber, confidence = self.predict(img)

        # Predict the test images
        self.predict_test_images()

        # Find images with the same number
        indices = np.where(self.testPredNumbers == predictedNumber)[0]
        imgs = self.testImages[indices]
        confidences = self.testConfidences[indices]

        # Sort the images by confidence
        sortedIndices = np.argsort(confidences)[-10:]
        return imgs[sortedIndices]
    
    def test_accuracy(self) -> float:
        """
        Test the accuracy of the SVM by predicting the number of the test images.

        @return: The accuracy of the SVM.
        """
        # Predict test images
        print("Computing accuracy on test images...")
        start = time.time()
        self.predict_test_images()

        # Compute the accuracy
        correct = self.testPredNumbers == self.testLabels
        accuracy = np.sum(correct) / len(correct)
        end = time.time()
        print(f"Done! Accuracy: {accuracy*100:.2f}%, Time: {end - start:.2f}s")
        return accuracy

    


def main_ssm():
    """
    Run some tests on finding similar images using the structural similarity measure.
    """

    app = App(mnist_path="./uncompressed_data")

    img = app.testImages[3450] # Arbitrary image from the set
    similar = app.findSimilar_ssim(img)

    fig, ax = plt.subplots(2, 6, figsize=(10, 4))
    ax[0, 0].imshow(img, cmap="gray")
    ax[0, 0].set_title(f"Original image")
    for i in range(5):
        ax[0, i+1].imshow(similar[i], cmap="gray")
    for i in range(5):
        ax[1, i+1].imshow(similar[i+5], cmap="gray")
    plt.show()

def main_svm():
    """
    Actual solution: Run some tests on finding similar images using the SVM.
    """
    app = App(mnist_path="./uncompressed_data")
    app.trainSVM()

    # Just for fun, will check the accuracy of the program.
    app.test_accuracy()

    img = app.testImages[3] # Arbitrary image from the set
    prediction, confidence = app.predict(img)
    print(f"Prediction: {prediction}, confidence: {confidence:.2f}")

    similar = app.findSimilar_svm(img)

    fig, ax = plt.subplots(2, 6, figsize=(10, 4))
    ax[0, 0].imshow(img, cmap="gray")
    ax[0, 0].set_title(f"Original (pred={prediction})")
    for i in range(5):
        ax[0, i+1].imshow(similar[i], cmap="gray")
    for i in range(5):
        ax[1, i+1].imshow(similar[i+5], cmap="gray")
    plt.show()

    # I have written a simple and fun example on a notebook to test the SVM.


if __name__ == "__main__":
    main_svm()
