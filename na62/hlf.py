import functools
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd

from . import constants
from .extract import cluster, photon_momentum, track

################################################################
# Three-vector operations
################################################################

def three_vectors_sum(vectors: List[pd.DataFrame]) -> pd.DataFrame:
    if len(vectors) == 0:
        return pd.Series()

    p = vectors[0]
    x = p["direction_x"]*p["momentum_mag"]
    y = p["direction_y"]*p["momentum_mag"]
    z = p["direction_z"]*p["momentum_mag"]
    for p in vectors[1:]:
        x += p["direction_x"]*p["momentum_mag"]
        y += p["direction_y"]*p["momentum_mag"]
        z += p["direction_z"]*p["momentum_mag"]

    mag = np.sqrt(x**2 + y**2 + z**2)
    x /= mag
    y /= mag
    z /= mag
    return pd.DataFrame({"direction_x": x, "direction_y": y, "direction_z": z, "momentum_mag": mag})


def three_vector_mag(vector: pd.DataFrame) -> pd.Series:
    return vector["momentum_mag"]


def three_vector_invert(vector: pd.DataFrame) -> pd.DataFrame:
    neg_vector = vector.copy()
    neg_vector[["direction_x", "direction_y", "direction_z"]] *= -1

    return neg_vector


################################################################
# Four-vector operations
################################################################

def four_vectors_sum(vectors: List[pd.DataFrame]) -> pd.DataFrame:
    if len(vectors) == 0:
        return pd.Series()

    momentum_sum = three_vectors_sum(vectors)

    p = vectors[0]
    momentum_sum["energy"] = p["energy"]
    for p in vectors[1:]:
        momentum_sum["energy"] += p["energy"]

    return momentum_sum


def four_vector_mag2(vector: pd.DataFrame) -> pd.DataFrame:
    return vector["energy"]**2 - three_vector_mag(vector)**2


def four_vector_mag(vector: pd.DataFrame) -> pd.DataFrame:
    mag2 = four_vector_mag2(vector)
    return np.sign(mag2)*np.sqrt(np.abs(mag2))


def four_vector_invert(vector: pd.DataFrame) -> pd.DataFrame:
    neg_vector = vector.copy()
    neg_vector[["direction_x", "direction_y", "direction_z", "energy"]] *= -1

    return neg_vector


################################################################
# Kinematic functions
################################################################

def invariant_mass(momenta: List[pd.DataFrame]) -> pd.Series:
    total_four_momentum = four_vectors_sum(momenta)
    return four_vector_mag(total_four_momentum)


def total_momentum(df: pd.DataFrame) -> pd.Series:
    t1 = track(df, 1).fillna(0)
    t2 = track(df, 2).fillna(0)
    t3 = track(df, 3).fillna(0)
    c1 = photon_momentum(df, 1).fillna(0)
    c2 = photon_momentum(df, 2).fillna(0)
    return three_vector_mag(three_vectors_sum([t1, t2, t3, c1, c2]))


def total_track_momentum(df: pd.DataFrame) -> pd.Series:
    t1 = track(df, 1).fillna(0)
    t2 = track(df, 2).fillna(0)
    t3 = track(df, 3).fillna(0)
    return three_vector_mag(three_vectors_sum([t1, t2, t3]))


def missing_mass_sqr(beam: pd.DataFrame, momenta: List[pd.DataFrame]) -> pd.Series:
    momenta_sum = four_vectors_sum(momenta)
    return four_vector_mag2(four_vectors_sum([beam, four_vector_invert(momenta_sum)]))


def missing_mass(beam: pd.DataFrame, momenta: List[pd.DataFrame]) -> pd.Series:
    momenta_sum = four_vectors_sum(momenta)
    return four_vector_mag(four_vectors_sum([beam, four_vector_invert(momenta_sum)]))


def propagate(track: pd.DataFrame, z_final: int, position_field_name: str = "position", direction_field_name: str = "direction") -> pd.DataFrame:
    dz = z_final - track[f"{position_field_name}_z"]
    factor = dz/track[f"{direction_field_name}_z"]
    pos_final_x = track[f"{position_field_name}_x"] + \
        track[f"{direction_field_name}_x"]*factor
    pos_final_y = track[f"{position_field_name}_y"] + \
        track[f"{direction_field_name}_y"]*factor
    pos_final_z = track[f"{position_field_name}_z"] + \
        track[f"{direction_field_name}_z"]*factor

    return pd.DataFrame({"position_x": pos_final_x, "position_y": pos_final_y, "position_z": pos_final_z})


################################################################
# Function to define and apply cuts
################################################################

def n(fun: Callable) -> Callable:
    def not_cut(df: pd.DataFrame) -> pd.Series:
        return ~fun(df)
    return not_cut


def combine_cuts(cuts: List[Callable]) -> Callable:
    '''
    Combine a list of cuts into a single Callable
    '''
    def cut(df: pd.DataFrame) -> pd.Series:
        all_cuts = map(lambda cut: cut(df), cuts)
        return functools.reduce(lambda c1, c2: c1 & c2, all_cuts)
    return cut


def select(df: pd.DataFrame, cuts: List[Callable]) -> pd.DataFrame:
    '''
    Apply a list of cuts to a dataframe
    '''
    return df.loc[combine_cuts(cuts)]


def identify(df: pd.DataFrame, definitions: Dict[str, List[Callable]]) -> pd.DataFrame:
    '''
    TODO
    '''
    ptype = pd.DataFrame(False, index=df.index,
                         columns=definitions.keys(), dtype=bool)
    for ptype_name in definitions:
        ptype.loc[combine_cuts(definitions[ptype_name])(df), ptype_name] = True
    return ptype


def make_eop_cut(min_eop: Union[float, None], max_eop: Union[float, None]) -> Callable:
    def cut(df: pd.DataFrame) -> pd.Series():
        min_cond = df["eop"] > min_eop if min_eop else True
        max_cond = df["eop"] < max_eop if max_eop else True
        return min_cond & max_cond
    return cut


def make_rich_cut(rich_hypothesis: Union[str, int], min_p: Union[None, int] = 15000, max_p: Union[None, int] = 40000) -> Callable:
    if isinstance(rich_hypothesis, str):
        rich_hypothesis = constants.rich_hypothesis_map[rich_hypothesis]
    momentum_condition = make_momentum_cut(min_p, max_p)

    def cut(df: pd.DataFrame) -> pd.Series:
        hypothesis = df["rich_hypothesis"] == rich_hypothesis
        return hypothesis & momentum_condition(df)
    return cut


def make_muv3_cut(has_muv3: bool, which_track: Union[None, str] = None) -> Callable:
    if which_track is None:
        which_track = ""
    else:
        which_track = f"{which_track}_"

    def cut(df: pd.DataFrame) -> pd.Series:
        return df[f"{which_track}has_muv3"] == has_muv3
    return cut


def make_momentum_cut(min_p: Union[None, int], max_p: Union[None, int], which_object: Union[None, str] = None) -> Callable:
    if which_object is None:
        which_object = ""
    else:
        which_object = f"{which_object}_"

    def cut(df: pd.DataFrame) -> pd.Series:
        serie_cut = df[f"{which_object}momentum_mag"]
        min_momentum_range = serie_cut > min_p if min_p else True
        max_momentum_range = serie_cut < max_p if max_p else True
        return min_momentum_range & max_momentum_range
    return cut


################################################################
# Other useful functions
################################################################

def lkr_energy(df: pd.DataFrame) -> pd.Series:
    t1 = track(df, 1)
    t2 = track(df, 2)
    t3 = track(df, 3)
    c1 = cluster(df, 1)
    c2 = cluster(df, 2)
    return t1["lkr_energy"].fillna(0) + t2["lkr_energy"].fillna(0) + t3["lkr_energy"].fillna(0) + c1["lkr_energy"].fillna(0) + c2["lkr_energy"].fillna(0)


def track_eop(df: pd.DataFrame, trackid: int) -> pd.Series:
    t = track(df, trackid)
    return t["lkr_energy"]/t["momentum_mag"]


def set_mass(df: pd.DataFrame, mass: float) -> pd.DataFrame:
    df["mass"] = mass
    df["energy"] = np.sqrt(mass**2 + df["momentum_mag"]**2)
    return df


def ring_radius(p, mass):
    n = 1.000063  # Refractive index in NA62
    f = 17*1000  # Focal lenght in NA62 (17m)
    c = 1        # Light speed in natural units

    # Compute the particle energy
    E = np.sqrt(mass**2 + p**2)
    # Compute the cos theta_c
    cost = (c*E)/(n*p)
    # Transform into radius using the focal length
    return np.arccos(cost) * f
