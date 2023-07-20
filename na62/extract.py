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
