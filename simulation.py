from __future__ import annotations

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
        self.pairs: set | None = None
        self.temps: np.ndarray | None = None
        self.tree: scipy.spatial.KDTree | None = None

    @abstractmethod
    def generate_packing(self, r: float) -> None:
        ...

    def _post_packing(self, r: float) -> None:
        """Work out the number of shapes packed and their radius

        Parameters
        ----------
        r : float
            radius of spheres
        """
        self.n = self.xi.shape[0]
        self.ri = np.asarray([r for _ in range(self.n)])

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

    def _set_source_sink_nodes(self) -> None:
        # find which nodes are sources and which are sinks
        min_height = self.xi[:, 1].min()
        max_height = self.xi[:, 1].max()
        self.source_nodes = np.argwhere(
            np.abs(self.xi[:, 1] - max_height) <= self.ri
        ).squeeze()
        self.sink_nodes = np.argwhere(
            np.abs(self.xi[:, 1]) <= min_height + self.ri
        ).squeeze()

    def generate_network(self) -> None:
        self._check_packing_created()
        self._set_source_sink_nodes()

        if self.tree is None:
            self.tree = scipy.spatial.KDTree(self.xi)
        self.pairs = set()
        max_r = self.ri.max()

        for i, pos in enumerate(self.xi):
            result = self.tree.query_ball_point(pos, 2 * max_r)
            for index in result:
                if index == i:
                    continue
                combined_radius = self.ri[i] + self.ri[index]
                if np.isclose(
                    np.linalg.norm(self.xi[i, :] - self.xi[index, :]),
                    combined_radius,
                    rtol=5e-2,
                ):
                    self.pairs.add((i, index))

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
        """Generate square matrix with the conductivity of each edge on the diagonal.

        Returns
        -------
        np.ndarray
            The conductivity matrix
        """
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


class EqualRadiusPacking(PackingMethod):
    def generate_network(self) -> None:
        """Create the network representation of the packing."""
        self._check_packing_created()
        self._set_source_sink_nodes()

        if self.tree is None:
            self.tree = scipy.spatial.KDTree(self.xi)
        self.pairs = self.tree.query_pairs(r=2 * self.ri[0], eps=1e-2)


class RegularPacking(EqualRadiusPacking):
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
        """Generates a regular packing between the limits.

        Parameters
        ----------
        r : float
            Radius of the spheres
        lim : tuple[float, float], optional
            Boundaries of domain to pack, by default (0, 1)
        """
        tmp = r + np.arange(lim[0], lim[1], 2 * r)
        self.xi = np.array(np.meshgrid(tmp, tmp)).T.reshape(-1, 2)


class OffsetRegularPacking(RegularPacking):
    def generate_packing(self, r: float) -> None:
        """Same as a regular packing however every odd row is offset by r to the right
        which makes the packing slightly denser.

        Parameters
        ----------
        r : float
            Radius of spheres
        """
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


def _insert_disks_at_points(im: np.ndarray, coords: np.ndarray, r: float) -> np.ndarray:
    """
    Insert disk of specified radius into an ND-image at given locations.
    """
    xlim, ylim = im.shape
    s = _make_disk(r)
    pt = coords.squeeze()
    for a, x in enumerate(range(pt[0] - r, pt[0] + r + 1)):
        if (x >= 0) and (x < xlim):
            for b, y in enumerate(range(pt[1] - r, pt[1] + r + 1)):
                if (y >= 0) and (y < ylim) and s[a, b] == 1:
                    im[x, y] = 0
    return im


def _make_disk(r: float) -> np.ndarray:
    """
    Generate a circular disk of the given radius
    """
    s = np.zeros((2 * r + 1, 2 * r + 1), dtype=type(r))
    for i in range(2 * r + 1):
        for j in range(2 * r + 1):
            if ((i - r) ** 2 + (j - r) ** 2) ** 0.5 <= r:
                s[i, j] = 1
    return s


class LowestPointFirstPacking(EqualRadiusPacking):
    def generate_packing(
        self,
        r: float,
        n_points: int = 1000,
        max_iter: int = 1000,
        debug: bool = False,
        random_state: np.random.Generator = None,
    ) -> None:
        """Similar to RSA however adds sphere in a random choice of the lowest possible
        options

        Parameters
        ----------
        r : float
            Radius of the sphere
        n_points : int, optional
            The number of points in each dimension to discretise the domain into,
            by default 1000.
            Increasing this will increase the run time but improve accuracy of
            packing.
        max_iter : int, optional
            The maximum number of iterations, by default 1000
        debug : bool, optional
            Displays the feasible addition space, by default False
        random_state : np.random.Generator, optional
            Random state to use when making the random choice out of the lowest points,
            by default None
        """
        if random_state is None:
            random_state = np.random.default_rng()
        self.xi = np.ndarray((0, 2))
        # scale the radius to the size of discretisation
        r = int(r * n_points)
        # lowest possible location of a sphere
        i_min = 0
        # initially all locations are viable for adding a sphere
        possible_locs = np.ones((n_points, n_points), dtype=bool)
        for _ in range(max_iter):
            if debug:
                # display the possible locations
                plt.imshow(possible_locs)
                plt.show()
            # indexes of all the possible place we could insert a sphere
            i, j = np.where(possible_locs[i_min : i_min + 2 * r, :])
            # if there is nowhere then we are done
            if len(i) == 0:
                break
            # Only want to consider those lowest locations
            options = np.where(i == i.min())[0]
            # Choose one at at random
            choice = random_state.choice(options)
            cen = np.vstack([i[choice] + i_min, j[choice]])
            # mask off area resulting from our choice
            possible_locs = _insert_disks_at_points(possible_locs, coords=cen, r=2 * r)
            self.xi = np.append(self.xi, cen.T / n_points, axis=0)
            i_min += i.min()

        self._post_packing(r / n_points)


class ClosestFirstPacking(EqualRadiusPacking):
    def generate_packing(
        self,
        r: float,
        n_points: int = 1000,
        max_iter: int = 1000,
        debug: bool = False,
        random_state: np.random.Generator = None,
    ) -> None:
        """Similar to RSA however adds sphere in a random choice so that it is
        touching another sphere.

        Parameters
        ----------
        r : float
            Radius of the sphere
        n_points : int, optional
            The number of points in each dimension to discretise the domain into,
            by default 1000.
            Increasing this will increase the run time but improve accuracy of
            packing.
        max_iter : int, optional
            The maximum number of iterations, by default 1000
        debug : bool, optional
            Displays the feasible addition space, by default False
        random_state : np.random.Generator, optional
            Random state to use when making the random choice out of the lowest points,
            by default None
        """
        if random_state is None:
            random_state = np.random.default_rng()
        self.xi = np.ndarray((0, 2))
        # scale the radius to the size of discretisation
        r = int(r * n_points)
        # initially all locations are viable for adding a sphere
        possible_locs = np.ones((n_points, n_points), dtype=bool)

        # need to insert the first sphere here so the distance transform works
        cen = random_state.choice(np.where(possible_locs), axis=1)
        # mask off area resulting from our choice
        possible_locs = _insert_disks_at_points(possible_locs, coords=cen, r=2 * r)
        self.xi = np.append(self.xi, np.atleast_2d(cen) / n_points, axis=0)

        for _ in range(max_iter):
            if debug:
                # display the possible locations
                plt.imshow(possible_locs)
                plt.show()
            # calculate distance from infeasible locations
            dist = scipy.ndimage.distance_transform_edt(possible_locs)

            # if everywhere is infeasible then we are done
            if not (dist != 0).any():
                break
            # get all the locations that are a minimum distance from infeasible region
            min_dist = dist[dist != 0].min()
            locs = np.asarray(np.where(dist == min_dist))
            # choose the next centre at random
            cen = random_state.choice(locs, axis=1)
            # mask off area resulting from our choice
            possible_locs = _insert_disks_at_points(possible_locs, coords=cen, r=2 * r)
            self.xi = np.append(self.xi, np.atleast_2d(cen) / n_points, axis=0)

        self._post_packing(r / n_points)


class RSAGrowthPacking(PackingMethod):
    def generate_packing(
        self,
        r: float,
        ncandidates: int = 1000,
        random_state: np.random.Generator = None,
    ) -> None:
        """Starts with RSA then grows spheres out. Doesn't work well with large
        a large starting radius.

        Parameters
        ----------
        r : float
            Initial radius of spheres
        ncandidates : int
            The number of candidate points to use per iteration of RSA
        random_state : np.random.Generator, optional
            Random state used by RSA, by default None
        """
        if random_state is None:
            random_state = np.random.default_rng()

        # generate initial random points
        engine = scipy.stats.qmc.PoissonDisk(
            d=2, radius=2 * r, ncandidates=ncandidates, seed=random_state
        )
        self.xi = engine.fill_space()
        # want to grow the radii in a random order
        random_state.shuffle(self.xi)
        self._post_packing(r)

        self.tree = scipy.spatial.KDTree(self.xi)

        for i, pos in enumerate(self.xi):
            # determine how much we can grow the radius by
            max_r = self.ri.max()
            result = np.asarray(self.tree.query_ball_point(pos, 4 * max_r))
            result = result[result != i]
            combined_radi = self.ri[i] + self.ri[result]
            dists = np.linalg.norm(self.xi[i, :] - self.xi[result, :], axis=1)
            self.ri[i] += (dists - combined_radi).min()


if __name__ == "__main__":
    p = RegularPacking()
    p.generate_packing(0.5 / 3)
    p.plot_packing()
    p.generate_network()
    p.plot_network()
