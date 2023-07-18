from typing import List
import pandas as pd
from extract import track, photon_momentum, cluster

def invariant_mass(df: List[pd.DataFrame]) -> pd.Series:
    pass

def total_momentum(df: pd.DataFrame) -> pd.Series:
    t1 = track(df, 1).fillna(0)
    t2 = track(df, 2).fillna(0)
    t3 = track(df, 3).fillna(0)
    c1 = photon_momentum(df, 1).fillna(0)
    c2 = photon_momentum(df, 2).fillna(0)
    return sum_momenta([t1, t2, t3, c1, c2])

def total_track_momentum(df: pd.DataFrame) -> pd.Series:
    t1 = track(df, 1).fillna(0)
    t2 = track(df, 2).fillna(0)
    t3 = track(df, 3).fillna(0)
    return sum_momenta([t1, t2, t3])

def sum_momenta(momenta: List[pd.DataFrame]) -> pd.Series:
    if len(momenta)==0:
        return pd.Series()

    p = momenta[0]
    x = p["direction_x"]*p["momentum_mag"]
    y = p["direction_y"]*p["momentum_mag"]
    z = p["direction_z"]*p["momentum_mag"]
    for p in momenta:
        x += p["direction_x"]*p["momentum_mag"]
        y += p["direction_y"]*p["momentum_mag"]
        z += p["direction_z"]*p["momentum_mag"]

    return np.sqrt(x**2 + y**2 + z**2)

def lkr_energy(df: pd.DataFrame) -> pd.Series:
    t1 = track(df, 1)
    t2 = track(df, 2)
    t3 = track(df, 3)
    c1 = cluster(df, 1)
    c2 = cluster(df, 2)
    return t1["lkr_energy"].fillna(0) + t2["lkr_energy"].fillna(0) + t3["lkr_energy"].fillna(0) + c1["lkr_energy"].fillna(0) + c2["lkr_energy"].fillna(0)

def compute_eop(df: pd.DataFrame, trackid: int) -> None:
    t = track(df, trackid)
    df[f"track{trackid}_eop"] = t["lkr_energy"]/t["momentum_mag"]
