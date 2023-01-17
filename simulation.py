from abc import ABC, abstractmethod
import random

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
        self.pairs: set | None = None
        self.temps: np.ndarray | None = None

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

        tree = scipy.spatial.KDTree(self.xi)
        self.pairs = tree.query_pairs(r=2 * self.ri[0], eps=1e-3)

    def _check_network_created(self) -> None:
        if self.pairs is None:
            raise ValueError("Need to generate network first.")

    def plot_network(self) -> None:
        """Plot network representation of the packing."""
        self._check_network_created()

        fig, ax = plt.subplots()
        # set the source and sink nodes to different colors
        colors = np.repeat("tab:blue", self.n).astype("object")
        colors[self.sink_nodes] = "tab:red"
        colors[self.source_nodes] = "tab:green"

        self._add_node_patches(ax, colors)

        # plot the network over the top of the patches
        for (i, j) in self.pairs:
            plt.plot(
                [self.xi[i, 0], self.xi[j, 0]], [self.xi[i, 1], self.xi[j, 1]], "-w"
            )
        ax.plot(self.xi[:, 0], self.xi[:, 1], "k.", markersize=10)
        plt.show()

    def _get_conductivity_matrix(self) -> np.ndarray:
        # TODO: use analytic formula!
        return np.eye(len(self.pairs), len(self.pairs))

    def solve_network(self, total_flux_in: float = 1.0) -> None:
        """Solve for the temperature in the network

        Parameters
        ----------
        total_flux_in : float
            The total amount of flux going into the material
        """
        self._check_network_created()
        graph = nx.from_edgelist(self.pairs)

        # create the oriented incidence matrix
        A = nx.incidence_matrix(graph, oriented=True).todense().T
        K = self._get_conductivity_matrix()

        LHS = A.T @ K @ A
        # flux into the material
        b = np.zeros((self.n, 1))
        b.put(self.source_nodes, total_flux_in / self.source_nodes.size)
        b = np.delete(b, self.sink_nodes, axis=0)
        temps = np.linalg.solve(LHS, b)

        # put the temps of the sink nodes back in
        self.temps = np.insert(temps, self.sink_nodes, 0)

    def _check_solved(self) -> None:
        if self.temps is None:
            raise ValueError("Network must be solved first")

    def plot_solution(self) -> None:
        """Plot the temperature of each node in the material"""
        self._check_solved()

        cmap = plt.colormaps["plasma"]
        colors = cmap(self.temps / np.amax(self.temps))

        fig, ax = plt.subplots()
        self._add_node_patches(ax, colors)
        fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap), ax=ax, label="Relative Temperature"
        )
        plt.show()


class RegularPacking(PackingMethod):
    def generate_packing(self, r: float) -> None:
        """Generates a regular packed grid of circles

        Parameters
        ----------
        r : float
            radius of circles
        """
        self._generate_packing(r)
        self._post_packing(r)

    def _generate_packing(self, r: float, lim: tuple[float, float] = (0, 1)) -> None:
        tmp = r + np.arange(lim[0], lim[1], 2 * r)
        self.xi = np.array(np.meshgrid(tmp, tmp)).T.reshape(-1, 2)

    def _post_packing(self, r: float) -> None:
        self.n = self.xi.shape[0]
        self.ri = np.asarray([r for _ in range(self.n)])


class OffsetRegularPacking(RegularPacking):
    def generate_packing(self, r: float) -> None:
        # start with the regular packing
        self._generate_packing(r, (-2 * r, 1 + r))

        # shift odd rows
        for i in range(-1, self.xi.shape[0], 4):
            self.xi[np.isclose(self.xi[:, 1], r * i), 0] += r

        # squash everything down
        self.xi[:, 1] -= (2 - np.sqrt(3)) * r * ((self.xi[:, 1] - r) / (2 * r))

        # trim off excess points
        self.xi = self.xi[
            (self.xi[:, 1] > 0) & (self.xi[:, 0] > -r) & (self.xi[:, 0] < 1 + r)
        ]

        self._post_packing(r)


class GravityPacking(PackingMethod):
    def generate_packing(self, r: float) -> None:
        self.xi = np.ndarray((0, 2))
        r = int(r * 100)
        x_min = r
        sites = np.ones((100, 100), dtype=bool)
        for _ in range(1000):
            x, y = np.where(sites[x_min : x_min + 2 * r, ...])
            if len(x) == 0:
                break
            options = np.where(x == x.min())[0]
            choice = np.random.randint(len(options))
            cen = np.vstack([x[options[choice]] + x_min, y[options[choice]]])
            sites = self._insert_disks_at_points(sites, coords=cen, r=2 * r)
            self.xi = np.append(self.xi, cen.T / 100, axis=0)
            x_min += x.min()

        self.n = self.xi.shape[0]
        self.ri = np.asarray([r / 100 for _ in range(self.n)])

    def _insert_disks_at_points(self, im, coords, r):
        """
        Insert disk of specified radius into an ND-image at given locations.
        """
        xlim, ylim = im.shape
        s = self._make_disk(r)
        pt = coords.squeeze()
        for a, x in enumerate(range(pt[0] - r, pt[0] + r + 1)):
            if (x >= 0) and (x < xlim):
                for b, y in enumerate(range(pt[1] - r, pt[1] + r + 1)):
                    if (y >= 0) and (y < ylim):
                        if s[a, b] == 1:
                            im[x, y] = 0
        return im

    def _make_disk(self, r):
        """
        Generate a circular disk of the given radius
        """
        s = np.zeros((2 * r + 1, 2 * r + 1), dtype=type(r))
        thresh = r - 0.001
        for i in range(2 * r + 1):
            for j in range(2 * r + 1):
                if ((i - r) ** 2 + (j - r) ** 2) ** 0.5 <= thresh:
                    s[i, j] = 1
        return s


if __name__ == "__main__":
    # # p = RegularPacking()
    # # p.generate_packing(0.1)
    # # p.plot_packing()
    # p = OffsetRegularPacking()
    # p.generate_packing(0.1)
    # p.plot_packing()
    # p.generate_network()
    # p.plot_network()
    # # p.solve_network()
    # # p.plot_solution()
    p = GravityPacking()
    p.generate_packing(0.1)
    p.plot_packing()
