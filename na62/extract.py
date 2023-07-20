from typing import List

import numpy as np
import pandas as pd

from .constants import lkr_position


def list_variables(df: pd.DataFrame) -> np.array:
    return df.columns.values


def print_event(df: pd.DataFrame, eventid: int) -> None:
    with pd.option_context('display.max_rows', None):
        print(df.loc[eventid])


def track(df: pd.DataFrame, trackID: int) -> pd.DataFrame:
    df = df.filter(like=f"track{trackID}")
    return df.rename(columns={_: _.replace(f"track{trackID}_", "") for _ in df})


def cluster(df: pd.DataFrame, clusterID: int) -> pd.DataFrame:
    df = df.filter(like=f"cluster{clusterID}")
    return df.rename(columns={_: _.replace(f"cluster{clusterID}_", "") for _ in df})


def photon_momentum(df: pd.DataFrame, clusterID: int) -> pd.DataFrame:
    c1 = cluster(df, clusterID)
    x = c1["position_x"] - df["vtx_x"]
    y = c1["position_y"] - df["vtx_y"]
    z = lkr_position - df["vtx_z"]
    mag = np.sqrt(x**2 + y**2 + z**2)
    c1["direction_x"] = x/mag
    c1["direction_y"] = y/mag
    c1["direction_z"] = z/mag
    c1["momentum_mag"] = c1["lkr_energy"]
    return c1


def all_tracks(df: pd.DataFrame) -> pd.DataFrame:
    t1 = track(df, 1)
    t2 = track(df, 2)
    t3 = track(df, 3)

    return pd.concat([t1.loc[t1["exists"]], t2.loc[t2["exists"]], t3.loc[t3["exists"]]])


def all_clusters(df: pd.DataFrame) -> pd.DataFrame:
    c1 = cluster(df, 1)
    c2 = cluster(df, 2)

    return pd.concat([c1.loc[c1["exists"]], c2.loc[c2["exists"]]])
