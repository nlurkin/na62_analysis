from typing import List

import pandas as pd
import numpy as np

from .extract import cluster, photon_momentum, track


def invariant_mass(df: List[pd.DataFrame]) -> pd.Series:
    pass


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


def missing_mass_sqr(beam: pd.DataFrame, tracks: List[pd.DataFrame]) -> pd.Series:
    beam_x = beam["direction_x"]*beam["momentum_mag"]
    beam_y = beam["direction_y"]*beam["momentum_mag"]
    beam_z = beam["direction_z"]*beam["momentum_mag"]
    beam_e = beam["energy"]

    decay_momentum_sum = four_vector_sum(tracks)
    decay_x = decay_momentum_sum["direction_x"] * \
        decay_momentum_sum["momentum_mag"]
    decay_y = decay_momentum_sum["direction_y"] * \
        decay_momentum_sum["momentum_mag"]
    decay_z = decay_momentum_sum["direction_z"] * \
        decay_momentum_sum["momentum_mag"]
    decay_e = decay_momentum_sum["energy"]

    return (beam_e - decay_e)**2 - (beam_x - decay_x)**2 - (beam_y - decay_y)**2 - (beam_z - decay_z)**2


def missing_mass(beam: pd.DataFrame, tracks: List[pd.DataFrame]) -> pd.Series:
    mm2 = missing_mass_sqr(beam, tracks)

    return np.sign(mm2)*np.sqrt(np.abs(mm2))


def four_vector_sum(vectors: List[pd.DataFrame]) -> pd.DataFrame:
    if len(vectors) == 0:
        return pd.Series()

    momentum_sum = three_vectors_sum(vectors)

    p = vectors[0]
    momentum_sum["energy"] = p["energy"]
    for p in vectors[1:]:
        momentum_sum["energy"] += p["energy"]

    return momentum_sum


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
