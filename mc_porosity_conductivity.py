from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from multiprocess import Pool
import pandas as pd

from simulation import (
    RegularPacking,
    OffsetRegularPacking,
    LowestPointFirstPacking,
    ClosestFirstPacking,
)


@dataclass
class PackingResult:
    porosity: float
    resistance: float


def _deterministic_wrapper(
    method, k: float, epsilon: float, radius: float
) -> PackingResult:
    p = method(k, epsilon)
    p.generate_packing(radius)
    p.generate_network()
    r = p.calculate_effective_resistance()
    porosity = p.calculate_porosity()
    return PackingResult(porosity, r)


def _nondeterministic_wrapper(
    method, k: float, epsilon: float, radius: float, repeats: int
) -> PackingResult:
    def wrapper():
        p = method(k, epsilon)
        p.generate_packing(radius, n_points=5000, max_iter=10000)
        p.generate_network()
        try:
            return p.calculate_effective_resistance(), p.calculate_porosity(
                n_points=5000
            )
        except ValueError:
            return np.nan, np.nan

    with Pool() as p:
        result = p.starmap(wrapper, [() for _ in range(repeats)])
    result = np.asarray(result)
    return PackingResult(np.nanmean(result[:, 1]), np.nanmean(result[:, 0]))


def mc_porosity_resistance(
    seed,
    radii_range: tuple[float, float],
    samples: int,
    non_deterministic_repeats: int,
    k: float,
    epsilon: float,
):
    rng = np.random.default_rng(seed)

    data = {
        "radius": [],
        "fixed_epsilon_regular_porosity": [],
        "fixed_epsilon_regular_resistance": [],
        "fixed_epsilon_offset_porosity": [],
        "fixed_epsilon_offset_resistance": [],
        "fixed_epsilon_lowest_first_porosity": [],
        "fixed_epsilon_lowest_first_resistance": [],
        "fixed_epsilon_closest_first_porosity": [],
        "fixed_epsilon_closest_first_resistance": [],
        "variable_epsilon_regular_porosity": [],
        "variable_epsilon_regular_resistance": [],
        "variable_epsilon_offset_porosity": [],
        "variable_epsilon_offset_resistance": [],
        "variable_epsilon_lowest_first_porosity": [],
        "variable_epsilon_lowest_first_resistance": [],
        "variable_epsilon_closest_first_porosity": [],
        "variable_epsilon_closest_first_resistance": [],
    }

    for i in range(samples):
        radius = rng.uniform(*radii_range)
        data["radius"].append(radius)

        # fixed epsilon
        reg = _deterministic_wrapper(RegularPacking, k, epsilon, radius)
        data["fixed_epsilon_regular_resistance"].append(reg.resistance)
        data["fixed_epsilon_regular_porosity"].append(reg.porosity)
        off = _deterministic_wrapper(OffsetRegularPacking, k, epsilon, radius)
        data["fixed_epsilon_offset_resistance"].append(off.resistance)
        data["fixed_epsilon_offset_porosity"].append(off.porosity)
        low = _nondeterministic_wrapper(
            LowestPointFirstPacking, k, epsilon, radius, non_deterministic_repeats
        )
        data["fixed_epsilon_lowest_first_resistance"].append(low.resistance)
        data["fixed_epsilon_lowest_first_porosity"].append(low.porosity)
        close = _nondeterministic_wrapper(
            ClosestFirstPacking, k, epsilon, radius, non_deterministic_repeats
        )
        data["fixed_epsilon_closest_first_resistance"].append(close.resistance)
        data["fixed_epsilon_closest_first_porosity"].append(close.porosity)

        # variable epsilon
        reg = _deterministic_wrapper(RegularPacking, k, radius / 100, radius)
        data["variable_epsilon_regular_resistance"].append(reg.resistance)
        data["variable_epsilon_regular_porosity"].append(reg.porosity)
        off = _deterministic_wrapper(OffsetRegularPacking, k, radius / 100, radius)
        data["variable_epsilon_offset_resistance"].append(off.resistance)
        data["variable_epsilon_offset_porosity"].append(off.porosity)
        low = _nondeterministic_wrapper(
            LowestPointFirstPacking, k, radius / 100, radius, non_deterministic_repeats
        )
        data["variable_epsilon_lowest_first_resistance"].append(low.resistance)
        data["variable_epsilon_lowest_first_porosity"].append(low.porosity)
        close = _nondeterministic_wrapper(
            ClosestFirstPacking, k, radius / 100, radius, non_deterministic_repeats
        )
        data["variable_epsilon_closest_first_resistance"].append(close.resistance)
        data["variable_epsilon_closest_first_porosity"].append(close.porosity)

        df = pd.DataFrame(data=data)
        df.to_csv(f"data_{i}.csv")


if __name__ == "__main__":
    mc_porosity_resistance(
        0x8C3C010CB4754C905776BDAC5EE7501,
        (0.04, 0.2),
        samples=20,
        non_deterministic_repeats=100,
        k=100,
        epsilon=1e-3,
    )
