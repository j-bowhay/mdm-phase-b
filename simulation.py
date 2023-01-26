from __future__ import annotations

import math
from functools import lru_cache

import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from multiprocess import Pool


@lru_cache
def _contact_resistance_expr(epsilon: float, b: float, k: float) -> float:
    """Calculates contact resistance in a sphere. For derivation see
    preliminary_investigations/Contact resistance calculations.ipynb

    Parameters
    ----------
    epsilon : float
        Lower limit of integration. Typically very small, control how large an area
        the spheres are in contact by.
    b : float
        How far into the sphere to integrate. Not critical to get this term right as
        epsilon << 1 will dominate.
    k : float
        The thermal conductivity (W/(K·m)) of the material

    Returns
    -------
    float
        The contact resistance in the sphere
    """
    return (1 / (4 * math.pi * k)) * math.log((2 * b) / epsilon - 1)


class PackingMethod:
    """
    Base class for generating packing in a unit square
    """

    def __init__(self, k: float, epsilon: float) -> None:
        """Initial the packing method.

        Parameters
        ----------
        k : float
            The thermal conductivity (W/(K·m)) of the material
        epsilon : float
            Lower limit of integration. Typically very small, control how large an
            area the spheres are in contact by.
            See preliminary_investigations/Contact resistance calculations.ipynb for
            more information
        """
        self.k = k
        self.epsilon = epsilon
        self.n: int | None = None
        self.xi: np.ndarray | None = None
        self.ri: np.ndarray | None = None
        self.source_nodes: np.ndarray | None = None
        self.sink_nodes: np.ndarray | None = None
        self.pairs: set | None = None
        self.temps: np.ndarray | None = None
        self.tree: scipy.spatial.KDTree | None = None

    def generate_packing(self, r: float) -> None:
        raise NotImplementedError

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

    def calculate_porosity(self, n_points: int = 1000) -> float:
        """Calculate the porosity of the packing.

        Parameters
        ----------
        n_points : int, optional
            Number of points in each dimension to use in the discretised, by default 1000

        Returns
        -------
        float
            The porosity of the packing
        """
        self._check_packing_created()

        domain = np.ones((n_points, n_points))
        for r, point in zip(
            (n_points * self.ri).astype(int), (n_points * self.xi).astype(int)
        ):
            domain = _insert_disks_at_points(domain, point, r)
        return (domain == 1).sum() / (n_points**2)

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
                    rtol=1e-2,
                ):
                    self.pairs.add((i, index))
        self.pairs = sorted(list(self.pairs))

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
        K = np.eye(len(self.pairs), len(self.pairs))
        for i, (start_node, end_node) in enumerate(self.pairs):
            K[i, i] = 1 / (
                _contact_resistance_expr(self.epsilon, self.ri[start_node] / 2, self.k)
                + _contact_resistance_expr(self.epsilon, self.ri[end_node] / 2, self.k)
            )
        return K

    def _get_incidence_matrix(self) -> np.ndarray:
        """Generates the oriented incidence matrix for the network

        Returns
        -------
        np.ndarray
            Oriented incidence matrix
        """
        graph = nx.from_edgelist(self.pairs)
        return (
            nx.incidence_matrix(
                graph,
                oriented=True,
                nodelist=sorted(
                    graph.nodes()
                ),  # needs sorting otherwise nx does something weird
                edgelist=self.pairs,
            )
            .todense()
            .T
        )

    def solve_network(self, total_flux_in: float = 1.0) -> None:
        """Solve for the temperature in the network

        Parameters
        ----------
        total_flux_in : float
            The total amount of flux going into the material
        """
        self._check_network_created()

        # create the oriented incidence matrix
        A = self._get_incidence_matrix()

        if np.linalg.matrix_rank(A) < self.n - 1:
            raise ValueError("Invalid packing (not connected)")

        K = self._get_conductivity_matrix()

        b = np.zeros(self.n)
        b.put(self.source_nodes, total_flux_in / self.source_nodes.size)

        # ground nodes
        A = np.delete(A, self.sink_nodes, axis=1)
        b = np.delete(b, self.sink_nodes)
        solved_temps = scipy.linalg.solve(A.T @ K @ A, b)

        self.temps = np.ones(self.n)
        self.temps[self.sink_nodes] = 0
        self.temps[self.temps != 0] = solved_temps

    def _check_solved(self) -> None:
        if self.temps is None:
            raise ValueError("Network must be solved first")

    def plot_solution(self) -> None:
        """Plot the temperature of each node in the material"""
        self._check_solved()

        cmap = plt.colormaps["plasma"]
        colors = cmap(self.temps / self.temps.max())

        fig, ax = plt.subplots()
        self._add_node_patches(ax, colors)
        scale = plt.cm.ScalarMappable(cmap=cmap)
        scale.set_clim(vmin=self.temps.min(), vmax=self.temps.max())
        fig.colorbar(scale, ax=ax, label="Relative Temperature")
        plt.show()

    def total_effective_resistance(self) -> float:
        """Calculates the total effective resistance of the network. This is the
        sum the sum of the effective resistance /between all distinct pairs of nodes.
        This is is related to the average power dissipation of the circuit with
        a random current excitation.

        References:
        https://web.stanford.edu/~boyd/papers/pdf/eff_res.pdf
        https://www.universiteitleiden.nl/binaries/content/assets/science/mi/scripties/ellensmaster.pdf
        https://cs-people.bu.edu/orecchia/CS591fa16/lecture7.pdf

        Returns
        -------
        float
            The total effective resistance.
        """
        self._check_network_created()

        K = self._get_conductivity_matrix()
        A = self._get_incidence_matrix()
        # Calculated the weighted graph laplacian
        L = A.T @ K @ A

        w = scipy.linalg.eig(L)[0]
        # Equation 15 from Ghosh, Boyd and Saberi
        return w.size * (1 / w).sum()


class EqualRadiusPacking(PackingMethod):
    def generate_network(self) -> None:
        """Create the network representation of the packing. Note we override the
        original method for creating the network here as it can be done more
        efficiently if all spheres are of equal radius.
        """
        self._check_packing_created()
        self._set_source_sink_nodes()

        if self.tree is None:
            self.tree = scipy.spatial.KDTree(self.xi)
        self.pairs = self.tree.query_pairs(r=2 * self.ri[0], eps=1e-2)
        self.pairs = sorted(list(self.pairs))


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


def _insert_disks_at_points(im: np.ndarray, coords: np.ndarray, r: int) -> np.ndarray:
    """
    Insert disk of specified radius into an array at given locations.
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


@lru_cache
def _make_disk(r: int) -> np.ndarray:
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

        # switch here due to matrix vs cartesian coords
        self.xi = np.flip(self.xi, axis=1)
        self._post_packing(r / n_points)


class LowestFirstFromDistributionPacking(PackingMethod):
    def generate_packing(
        self,
        r_distribution: scipy.stats.rv_continuous,
        n_points: int = 1000,
        max_iter: int = 1000,
        debug: bool = False,
        random_state: np.random.Generator = None,
    ) -> None:
        """Similar packing method to lowest first but the radii are sampled from a
        distribution.

        Parameters
        ----------
        r_distribution : scipy.stats.rv_continuous
            The distribution from which the radii are sampled.
        n_points : int, optional
            The number of points in each dimension to discretise the domain into,
            by default 1000.
            Increasing this will increase the run time but improve accuracy of
            packing.
        max_iter : int, optional
            The maximum number of iterations, by default 1000
        debug : bool, optional
            Displays some plot to help with debugging, by default False
        random_state : np.random.Generator, optional
            The random state to use, by default None

        Raises
        ------
        ValueError
            If the sampled radius is negative
        """
        if random_state is None:
            random_state = np.random.default_rng()
        self.xi = np.ndarray((0, 2))
        self.ri = np.ndarray((0))
        # lowest possible location of a sphere
        for i in range(max_iter):
            # sample the radius from the distribution
            r = r_distribution.rvs(1)
            if r < 0:
                raise ValueError("Radius cannot be negative")
            r_scaled = int(r * n_points)
            # generate feasible region
            possible_locs = np.ones((n_points, n_points), dtype=bool)
            for r_circ, cen in zip(
                (n_points * self.ri).astype(int), (n_points * self.xi).astype(int)
            ):
                possible_locs = _insert_disks_at_points(
                    possible_locs, cen, r_scaled + r_circ
                )
            if debug:
                plt.imshow(possible_locs)
            # indexes of all the possible place we could insert a sphere
            i, j = np.where(possible_locs)
            # if there is nowhere then we are done
            if len(i) == 0:
                break
            # Only want to consider those lowest locations
            options = np.where(i == i.min())[0]
            # Choose one at at random
            choice = random_state.choice(options)
            cen = np.vstack([i[choice], j[choice]])
            if debug:
                plt.plot(cen[1], cen[0], "r*", markersize=20)
                plt.show()
            self.xi = np.append(self.xi, cen.T / n_points, axis=0)
            self.ri = np.append(self.ri, r, axis=0)

        # switch here due to matrix vs cartesian coords
        self.xi = np.flip(self.xi, axis=1)
        self.n = self.ri.size


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


def get_porosity_distribution(
    method: PackingMethod, radius: float, number_of_trials: int
) -> list[float]:
    """Samples the porosity of a packing method lots of times to get the distribution of porosity.

    Parameters
    ----------
    method : PackingMethod
        The packing class to use
    number_of_trials : int
        Number of times to generate a packing and get the porosity

    Returns
    -------
    list[float]
        The sampled porosities
    """

    def wrapper() -> float:
        # can use any k and epsilon here it doesn't matter
        p = method(0, 0)
        p.generate_packing(radius)
        return p.calculate_porosity()

    with Pool() as p:
        return list(p.starmap(wrapper, [() for _ in range(number_of_trials)]))


if __name__ == "__main__":
    # p = LowestFirstFromDistributionPacking(100, 1e-3)
    # p.generate_packing(scipy.stats.gamma(10, scale=0.05 / 4), n_points=1000)
    p = RegularPacking(100, 1e-3)
    p.generate_packing(0.5/10)
    p.plot_packing()
    p.generate_network()
    print(p.total_effective_resistance())
    # p.plot_network()
    # p.solve_network()
    # p.plot_solution()
    # print(f"Packing porosity: {p.calculate_porosity()}")
    # plt.hist(get_porosity_distribution(ClosestFirstPacking, 0.5 / 4, 100))
    # plt.show()
