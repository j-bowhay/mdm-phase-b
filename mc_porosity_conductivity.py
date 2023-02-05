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
    conductivity: float
    conductivity_sd: float


def _deterministic_wrapper(
    method, k: float, epsilon: float, radius: float
) -> PackingResult:
    p = method(k, epsilon)
    p.generate_packing(radius)
    p.generate_network()
    r = p.calculate_effective_conductivity(1)
    porosity = p.calculate_porosity()
    return PackingResult(porosity, r, np.nan)


def _nondeterministic_wrapper(
    method, k: float, epsilon: float, radius: float, repeats: int
) -> PackingResult:
    def wrapper():
        p = method(k, epsilon)
        p.generate_packing(radius, n_points=6000, max_iter=10000)
        p.generate_network()
        try:
            return p.calculate_effective_conductivity(1), p.calculate_porosity(
                n_points=2000
            )
        except ValueError:
            return PackingResult(np.nan, np.nan, np.nan)

    with Pool() as p:
        result = p.starmap(wrapper, [() for _ in range(repeats)])
    result = np.asarray(result)
    return PackingResult(np.nanmean(result[:, 1]), np.nanmean(result[:, 0]), np.std(result[:, 0]))


def mc_porosity_resistance(
    seed,
    radii_range: tuple[float, float],
    samples: int,
    non_deterministic_repeats: int,
    k: float,
):
    rng = np.random.default_rng(seed)

    data = {
        "radius": [],
        "variable_epsilon_regular_porosity": [],
        "variable_epsilon_regular_conductivity": [],
        "variable_epsilon_offset_porosity": [],
        "variable_epsilon_offset_conductivity": [],
        "variable_epsilon_lowest_first_porosity": [],
        "variable_epsilon_lowest_first_conductivity": [],
        "variable_epsilon_lowest_first_conductivity_sd": [],
        "variable_epsilon_closest_first_porosity": [],
        "variable_epsilon_closest_first_conductivity": [],
        "variable_epsilon_closest_first_conductivity_sd": [],
    }

    for i, radius in enumerate(np.linspace(*radii_range, num=samples)):
        print(i)
        data["radius"].append(radius)
        epsilon = radius/100

        reg = _deterministic_wrapper(RegularPacking, k, epsilon, radius)
        data["variable_epsilon_regular_conductivity"].append(reg.conductivity)
        data["variable_epsilon_regular_porosity"].append(reg.porosity)
        off = _deterministic_wrapper(OffsetRegularPacking, k, epsilon, radius)
        data["variable_epsilon_offset_conductivity"].append(off.conductivity)
        data["variable_epsilon_offset_porosity"].append(off.porosity)
        low = _nondeterministic_wrapper(
            LowestPointFirstPacking, k, epsilon, radius, non_deterministic_repeats
        )
        data["variable_epsilon_lowest_first_conductivity"].append(low.conductivity)
        data["variable_epsilon_lowest_first_porosity"].append(low.porosity)
        data["variable_epsilon_lowest_first_conductivity_sd"].append(low.conductivity_sd)
        close = _nondeterministic_wrapper(
            ClosestFirstPacking, k, epsilon, radius, non_deterministic_repeats
        )
        data["variable_epsilon_closest_first_conductivity"].append(close.conductivity)
        data["variable_epsilon_closest_first_porosity"].append(close.porosity)
        data["variable_epsilon_closest_first_conductivity_sd"].append(close.conductivity_sd)

        df = pd.DataFrame(data=data)
        df.to_csv(f"data_{i}.csv")


if __name__ == "__main__":
    mc_porosity_resistance(
        0x8C3C010CB4754C905776BDAC5EE7501,
        (0.04, 0.2),
        samples=30,
        non_deterministic_repeats=20,
        k=100,
    )
