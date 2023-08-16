import functools
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd

from . import constants
from .extract import cluster, photon_momentum, track, get_beam


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


def missing_mass_sqr(df: pd.DataFrame, momenta_or_masses: Union[Dict[str, float], List[pd.DataFrame]]) -> pd.Series:
    if isinstance(momenta_or_masses, dict):
        return missing_mass_sqr_from_fulldf(df, momenta_or_masses)
    else:
        return missing_mass_sqr_from_4vector(df, momenta_or_masses)


def missing_mass_sqr_from_4vector(beam: pd.DataFrame, momenta: List[pd.DataFrame]) -> pd.Series:
    momenta_sum = four_vectors_sum(momenta)
    return four_vector_mag2(four_vectors_sum([beam, four_vector_invert(momenta_sum)]))


def missing_mass_sqr_from_fulldf(df: pd.DataFrame, mass_assignments: Dict[str, float]) -> pd.Series:
    track_masses = {}
    cluster_masses = {}
    for trackID in np.arange(1, 4):
        if f"track{trackID}" in mass_assignments:
            track_masses[trackID] = mass_assignments[f"track{trackID}"]
    for clusterID in np.arange(1, 3):
        if f"cluster{clusterID}" in mass_assignments:
            cluster_masses[clusterID] = mass_assignments[f"cluster{clusterID}"]

    # Extract the beam
    beam = set_mass(get_beam(df), constants.kaon_charged_mass)

    # Extract the tracks and photons based on mass_assignments
    objects = []
    for trackID in track_masses:
        objects.append(set_mass(track(df, trackID), track_masses[trackID]))
    for clusterID in cluster_masses:
        objects.append(set_mass(photon_momentum(
            df, clusterID), cluster_masses[clusterID]))

    return missing_mass_sqr_from_4vector(beam, objects)


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


def make_eop_cut(min_eop: Union[float, None], max_eop: Union[float, None], which_track: Union[None, str] = None) -> Callable:
    which_track = _select_object(which_track)

    def cut(df: pd.DataFrame) -> pd.Series():
        min_cond = df[f"{which_track}eop"] > min_eop if min_eop else True
        max_cond = df[f"{which_track}eop"] < max_eop if max_eop else True
        return min_cond & max_cond
    return cut


def make_rich_cut(rich_hypothesis: Union[str, int],
                  min_p: Union[None, int] = 15000, max_p: Union[None, int] = 40000,
                  which_track: Union[None, str] = None) -> Callable:
    if isinstance(rich_hypothesis, str):
        rich_hypothesis = constants.rich_hypothesis_map[rich_hypothesis]
    momentum_condition = make_momentum_cut(min_p, max_p)
    which_track = _select_object(which_track)

    def cut(df: pd.DataFrame) -> pd.Series:
        hypothesis = df[f"{which_track}rich_hypothesis"] == rich_hypothesis
        return hypothesis & momentum_condition(df)
    return cut


def make_muv3_cut(has_muv3: bool, which_track: Union[None, str] = None) -> Callable:
    which_track = _select_object(which_track)

    def cut(df: pd.DataFrame) -> pd.Series:
        return df[f"{which_track}has_muv3"] == has_muv3
    return cut


def make_momentum_cut(min_p: Union[None, int], max_p: Union[None, int], which_object: Union[None, str] = None) -> Callable:
    which_object = _select_object(which_object)

    def cut(df: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        if isinstance(df, pd.DataFrame):
            serie_cut = df[f"{which_object}momentum_mag"]
        else:
            serie_cut = df
        min_momentum_range = serie_cut > min_p if min_p else True
        max_momentum_range = serie_cut < max_p if max_p else True
        return min_momentum_range & max_momentum_range
    return cut


def make_total_momentum_cut(min_p: Union[None, int], max_p: Union[None, int] = None) -> Callable:
    momentum_condition = make_momentum_cut(min_p, max_p)

    def cut(df: pd.DataFrame) -> pd.Series:
        p_tot = total_momentum(df)
        return momentum_condition(p_tot)
    return cut


def make_missing_mass_sqr_cut(min_mm2: Union[None, float], max_mm2: Union[None, float], mass_assignments: Dict[str, float]) -> Callable:
    def cut(df: pd.DataFrame) -> pd.Series:
        mmass_sqr = missing_mass_sqr(df, mass_assignments)

        min_mmass_sqr = mmass_sqr > min_mm2 if min_mm2 else True
        max_mmass_sqr = mmass_sqr < max_mm2 if max_mm2 else True

        return min_mmass_sqr & max_mmass_sqr
    return cut


def make_exists_cut(exists: Union[None, List[str]], not_exists: Union[None, List[str]]) -> Callable:
    if not exists:
        exists = []
    if not not_exists:
        not_exists = []

    def cut(df: pd.DataFrame) -> pd.Series:
        this_cut = pd.Series(True, index=df.index, dtype=bool)
        for name in exists:
            this_cut &= df[name]
        for name in not_exists:
            this_cut &= ~df[name]
        return this_cut
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


################################################################
# Internal functions
################################################################

def _select_object(which_object: Union[None, str]):
    if which_object is None:
        return ""
    else:
        return f"{which_object}_"
