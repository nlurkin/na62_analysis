"""
This module contains function that extract parts of an original dataframe in "Full" format into a format that is
more useful for manipulation of the individual parts.
"""


import numpy as np
import pandas as pd

from .constants import lkr_position


def print_event(df: pd.DataFrame, eventid: int) -> None:
    """
    Print the event with the given index without ellipsis

    :param df: Dataframe
    :param eventid: index of the event in the dataframe
    """
    with pd.option_context('display.max_rows', None):
        print(df.loc[eventid])


def track(df: pd.DataFrame, trackID: int) -> pd.DataFrame:
    """
    Extract all the variables related to the specified track from the dataframe in "Full" format.

    :param df: Full dataframe
    :param trackID: ID of the track to be extracted (1, 2 or 3)
    :return: A new 'track' dataframe where the `track{trackID}_` prefix has been removed
    """

    df = df.filter(like=f"track{trackID}")
    return df.rename(columns={_: _.replace(f"track{trackID}_", "") for _ in df})


def cluster(df: pd.DataFrame, clusterID: int) -> pd.DataFrame:
    """
    Extract all the variables related to the specified cluster from the dataframe in "Full" format.

    :param df: Full dataframe
    :param clusterID: ID of the cluster to be extracted (1 or 2)
    :return: A new 'cluster' dataframe where the `cluster{clusterID}_` prefix has been removed
    """
    df = df.filter(like=f"cluster{clusterID}")
    return df.rename(columns={_: _.replace(f"cluster{clusterID}_", "") for _ in df})


def photon_momentum(df: pd.DataFrame, clusterID: int) -> pd.DataFrame:
    """
    Extract a cluster from the "Full" dataframe and compute the momentum variables:
     - direction_{x,y,z} = direction between position on LKr and vertex
     - momentum_mag = LKr Energy

    :param df: Full dataframe
    :param clusterID: ID of the cluster to extract as momentum
    :return: A new "momentum" dataframe
    """
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
    """
    Extract all existing tracks from the "Full" dataframe and concatenate them in a single 'track' dataframe

    :param df: Full dataframe
    :return: A new concatenated 'track' dataframe containing all the existing tracks in the original dataframe
    """
    t1 = track(df, 1)
    t2 = track(df, 2)
    t3 = track(df, 3)

    return pd.concat([t1.loc[t1["exists"]], t2.loc[t2["exists"]], t3.loc[t3["exists"]]]).reset_index()


def all_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all existing cluster from the "Full" dataframe and concatenate them in a single 'cluster' dataframe

    :param df: Full dataframe
    :return: A new concatenated 'cluster' dataframe containing all the existing clusters in the original dataframe
    """
    c1 = cluster(df, 1)
    c2 = cluster(df, 2)

    return pd.concat([c1.loc[c1["exists"]], c2.loc[c2["exists"]]])


def get_beam(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the beam variables from the "Full" dataframe.

    :param df: Full dataframe
    :return: A new 'beam' dataframe where the `beam_` prefix has been removed
    """

    beam = df.filter(like="beam_")
    beam = beam.rename(columns={_: _.replace("beam_", "")
                       for _ in beam.columns})
    return beam
