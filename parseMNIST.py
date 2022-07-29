import struct
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def loadMNIST(folder_path:str = ".") -> Tuple[np.ndarray]:
    """
    Read MNIST data from disk and parse it into arrays. folder_path can be specified if the data is not in the current directory.

    Four documents are to be read:
    - train-images.idx3-ubyte: training images
    - train-labels.idx1-ubyte: training labels
    - t10k-images.idx3-ubyte: test images
    - t10k-labels.idx1-ubyte: test labels

    @param folder_path: Path to the folder containing the MNIST data.

    @return: A tuple containing the training images, training labels, test images, and test labels as numpy arrays.
    """

    # Read train images
    with open(f"{folder_path}/train-images.idx3-ubyte", "rb") as doc:
        # Read information in first 16 bytes (>IIII meaning 4 unsigned ints in big endian, or MSB first):
        _, size, rows, cols = struct.unpack(">IIII", doc.read(16))
        trainImages = np.fromfile(doc, dtype=np.uint8) # Read the rest of the file as a 1D array of unsigned bytes
        trainImages = trainImages.reshape((size, rows, cols)) # Reshape such that first dimension is the number of images
    
    # Read train labels
    with open(f"{folder_path}/train-labels.idx1-ubyte", "rb") as doc:
        _, size = struct.unpack(">II", doc.read(8))
        trainLabels = np.fromfile(doc, dtype=np.uint8)
    
    # Read test images
    with open(f"{folder_path}/t10k-images.idx3-ubyte", "rb") as doc:
        _, size, rows, cols = struct.unpack(">IIII", doc.read(16))
        testImages = np.fromfile(doc, dtype=np.uint8)
        testImages = testImages.reshape((size, rows, cols))

    # Read test labels
    with open(f"{folder_path}/t10k-labels.idx1-ubyte", "rb") as doc:
        _, size = struct.unpack(">II", doc.read(8))
        testLabels = np.fromfile(doc, dtype=np.uint8)
    
    return trainImages, trainLabels, testImages, testLabels

def main():
    """
    Read MNIST data and show some examples.
    """
    # I saved the uncompressed data in a folder named "uncompressed_data", change the path if necessary (empty means current directory)
    trainImages, trainLabels, testImages, testLabels = loadMNIST("./uncompressed_data")
    print(trainImages.shape, trainLabels.shape, testImages.shape, testLabels.shape)

    fig, ax = plt.subplots(2, 2, figsize=(10, 4))
    ax[0, 0].imshow(trainImages[0], cmap="gray")
    ax[0, 0].set_title(f"(Train) Label: {trainLabels[0]}")
    ax[0, 1].imshow(trainImages[1], cmap="gray")
    ax[0, 1].set_title(f"(Train) Label: {trainLabels[1]}")
    ax[1, 0].imshow(testImages[0], cmap="gray")
    ax[1, 0].set_title(f"(Test) Label: {testLabels[0]}")
    ax[1, 1].imshow(testImages[1], cmap="gray")
    ax[1, 1].set_title(f"(Test) Label: {testLabels[1]}")
    plt.show()


if __name__ == "__main__":
    main()
