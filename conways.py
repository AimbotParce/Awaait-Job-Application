import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve


def GoL_step(gameState: np.ndarray) -> np.ndarray:
    """
    Computes the next starte of Conway's Game of Life based on the current screen state.

    @param gameState: Numpy matrix as the current game state. Must contain values of either 0 (dead) or 1 (alive).

    @return: Numpy matrix as the next game state.
    """
    # Start by counting all the neighbours
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbourCount = convolve(gameState, kernel, mode="constant", cval=0)

    # Compute the next state
    # Rules: For every cell and their neighbours:
    # - If dead, and exactly 3 neighbours, become alive
    # - If alive, and 2 or 3 neighbours, stay alive
    # - Otherwise, stay dead (or die)
    nextState = np.zeros(gameState.shape)
    nextState[(neighbourCount == 3) & (gameState == 0)] = 1
    nextState[((neighbourCount == 2) | (neighbourCount == 3)) & (gameState == 1)] = 1

    return nextState


def main():
    """
    Run some examples of Game of Life steps.
    """

    # Test some game states
    test1 = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    test2 = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0]])
    # Compute next steps and show them
    print(f"{'-'*15}\nTest1:\n", GoL_step(test1))
    print(f"{'-'*15}\nTest2:\n", GoL_step(test2))

    # A more fun experiment:
    test3 = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    # Animate the game for 20 iterations
    plt.figure()
    plt.ion()

    for _ in range(20):
        plt.clf()
        plt.imshow(test3)
        test3 = GoL_step(test3)
        plt.draw()
        plt.pause(0.1)

    # We can see not only it is working, but it is also stable and fast.


if __name__ == "__main__":
    main()
