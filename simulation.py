from abc import ABC, abstractmethod

import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class PackingMethod(ABC):
    """
    Base class for generating packing in a unit square
    """

    def __init__(self) -> None:
        self.n: int | None = None
        self.xi: np.ndarray | None = None
        self.ri: np.ndarray | None = None
        self.source_nodes: np.ndarray | None = None
        self.sink_nodes: np.ndarray | None = None

    @abstractmethod
    def generate_packing(self, r: float) -> None:
        ...

    def _add_node_patches(self, ax: plt.Axes, colors: np.ndarray | None = None) -> None:
        """Adds circle patches to an axes at the location of each of the particles.

        Parameters
        ----------
        ax : plt.Axes
            The axes to add the patches too.
        colors : np.ndarray | None, optional
            Array of colors for each of the patches, by default None
        """
        if colors is None:
            colors = np.repeat("tab:blue", self.ri.size)
        for r, point, color in zip(self.ri, self.xi, colors):
            ax.add_patch(patches.Circle(point, r, color=color))
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set_aspect("equal")
    
    def _check_packing_created(self) -> None:
        if self.xi is None:
            raise ValueError("Need to generate packing first.")

    def plot_packing(self) -> None:
        """Plot the basic packing."""
        self._check_packing_created
        fig, ax = plt.subplots()
        self._add_node_patches(ax)
        plt.show()

    def generate_network(self) -> None:
        """Create the network representation of the packing."""
        self._check_packing_created()
        
        # find which nodes are sources and which are sinks
        self.source_nodes = np.argwhere(np.abs(self.xi[:, 1] - 1) <= self.ri).squeeze()
        self.sink_nodes = np.argwhere(np.abs(self.xi[:, 1]) <= self.ri).squeeze()
        
        adjacency = np.zeros((self.n, self.n))
        tree = scipy.spatial.KDTree(self.xi)

    def plot_network(self) -> None:
        """Plot network representation of the packing."""
        fig, ax = plt.subplots()
        colors = np.repeat("tab:blue", self.n).astype("object")
        colors[self.sink_nodes] = "tab:red"
        colors[self.source_nodes] = "tab:green"
        self._add_node_patches(ax, colors)
        ax.plot(self.xi[:, 0], self.xi[:, 1], "k.", markersize=10)
        plt.show()


class RegularPacking(PackingMethod):
    def generate_packing(self, r: float) -> None:
        """Generates a regular packed grid of circles

        Parameters
        ----------
        r : float
            radius of circles
        """
        self.n = int(1 / (2 * r)) ** 2

        tmp = np.arange(r, 1, 2 * r)
        self.xi = np.array(np.meshgrid(tmp, tmp)).T.reshape(-1, 2)
        self.ri = np.asarray([r for _ in range(self.n)])


if __name__ == "__main__":
    p = RegularPacking()
    p.generate_packing(0.05)
    p.plot_packing()
    p.generate_network()
    p.plot_network()
