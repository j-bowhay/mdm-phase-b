from abc import ABC, abstractmethod

import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class PackingMethod(ABC):
    """
    Base class for generating packing in a unit square
    """

    def __init__(self) -> None:
        self.n: int = None
        self.xi: np.ndarray = None
        self.ri: np.ndarray = None

    @abstractmethod
    def generate_packing(self, r: float) -> None:
        ...

    def plot_packing(self) -> None:
        fig, ax = plt.subplots()

        for r, point in zip(self.ri, self.xi):
            ax.add_patch(patches.Circle(point, r))

        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set_aspect("equal")
        plt.show()


class RegularPacking(PackingMethod):
    def generate_packing(self, r: float) -> None:
        self.n = int(1 / (2 * r)) ** 2

        tmp = np.arange(r, 1, 2 * r)
        self.xi = np.array(np.meshgrid(tmp, tmp)).T.reshape(-1, 2)
        self.ri = np.asarray([r for _ in range(self.n)])


if __name__ == "__main__":
    p = RegularPacking()
    p.generate_packing(0.05)
    p.plot_packing()
