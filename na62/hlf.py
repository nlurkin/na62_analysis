from typing import List

import pandas as pd
import numpy as np

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
    pos_final_x = track[f"{position_field_name}_x"] + track[f"{direction_field_name}_x"]*factor
    pos_final_y = track[f"{position_field_name}_y"] + track[f"{direction_field_name}_y"]*factor
    pos_final_z = track[f"{position_field_name}_z"] + track[f"{direction_field_name}_z"]*factor

    return pd.DataFrame({"position_x": pos_final_x, "position_y": pos_final_y, "position_z": pos_final_z})

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
    n = 1.000063 # Refractive index in NA62
    f = 17*1000  # Focal lenght in NA62 (17m)
    c = 1        # Light speed in natural units

    # Compute the particle energy
    E = np.sqrt(mass**2 + p**2)
    # Compute the cos theta_c
    cost = (c*E)/(n*p)
    # Transform into radius using the focal length
    return np.arccos(cost) * f