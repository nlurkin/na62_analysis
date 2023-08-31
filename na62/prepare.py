from typing import Union

import numpy as np
import pandas as pd
import uproot

from .hlf import track_eop


def import_root_file(filename: str, limit: Union[None, int] = None) -> pd.DataFrame:
    """Read a ROOT file and import the NA62Flat TTree into a pandas DataFrame. Some pre-processing is performed on the dataframe
    to clean it and pre-compute some derived values.

    :param filename: Path to the ROOT file to load
    :param limit: Maximum number of events to load. If None, load the complete TTree. (default None)
    :return: Full dataframe
    """
    with uproot.open(filename) as fd:
        x = fd.get("export_flat/NA62Flat")
        data = x.arrays(x.keys(), library="pd", entry_stop=limit)
        data = data.replace([np.inf, -np.inf], np.nan)
        type_dict = {"beam_momentum_mag": np.float64, "beam_direction_x": np.float64,
                     "beam_direction_y": np.float64, "beam_direction_z": np.float64}
        for trackid in range(1, 4):
            type_dict[f"track{trackid}_momentum_mag"] = np.float64
            type_dict[f"track{trackid}_direction_x"] = np.float64
            type_dict[f"track{trackid}_direction_y"] = np.float64
            type_dict[f"track{trackid}_direction_z"] = np.float64
            type_dict[f"track{trackid}_direction_am_x"] = np.float64
            type_dict[f"track{trackid}_direction_am_y"] = np.float64
            type_dict[f"track{trackid}_direction_am_z"] = np.float64

        data = data.astype(type_dict)
        clean_clusters(data)
        clean_tracks(data)

        compute_derived(data)
        # Update with real value - either limit itself, or less if there was not so much data to read
        limit = len(data)
        normalization = sample_normalization(fd, limit)
        data.attrs["acceptances"] = pd.Series([limit/normalization if normalization!=0 else np.nan], index=["pre-selection"])
    return data, normalization


def import_root_file_mc_truth(filename: str, limit: Union[None, int] = None) -> pd.DataFrame:
    """Read a ROOT file and import the NA62MCFlat TTree into a pandas DataFrame. Some pre-processing is performed on the dataframe
    to clean it and pre-compute some derived values.

    :param filename: Path to the ROOT file to load
    :param limit: Maximum number of events to load. If None, load the complete TTree. (default None)
    :return: Full dataframe
    """

    with uproot.open(filename) as fd:
        x = fd.get("export_flat/NA62MCFlat")
        data = x.arrays(x.keys(), library="pd", entry_stop=limit)
        beam_vars = data.filter(like="track0").columns
        data = data.rename(columns={_: _.replace(
            "track0", "beam") for _ in beam_vars})
        data = data.replace([np.inf, -np.inf], np.nan)
        type_dict = {"beam_momentum_mag": np.float64, "beam_direction_x": np.float64,
                     "beam_direction_y": np.float64, "beam_direction_z": np.float64}
        for trackid in range(1, 4):
            type_dict[f"track{trackid}_momentum_mag"] = np.float64
            type_dict[f"track{trackid}_direction_x"] = np.float64
            type_dict[f"track{trackid}_direction_y"] = np.float64
            type_dict[f"track{trackid}_direction_z"] = np.float64
            type_dict[f"track{trackid}_direction_am_x"] = np.float64
            type_dict[f"track{trackid}_direction_am_y"] = np.float64
            type_dict[f"track{trackid}_direction_am_z"] = np.float64

        data = data.astype(type_dict)
        clean_tracks(data)

    return data


def import_root_files(filenames: list[str], total_limit: Union[None, int] = None, file_limit: Union[None, int] = None) -> pd.DataFrame:
    """Read a list of ROOT file and import the NA62Flat TTree into a single, merged, pandas DataFrame. Some pre-processing
    is performed on the dataframe to clean it and pre-compute some derived values.

    :param filenames: List of paths to ROOT files to load
    :param total_limit: Absolute maximum number of events to load. If None, no limit is applied. (default None)
    :param file_limit: Maximum number of events to load for each file. If None, no limit is applied.
        This can be combined with the 'total_limit' parameter. (default None)
    :return: Full dataframe
    """
    data_list = []
    total_normalization = 0
    for filename in filenames:
        curr_limit = min(file_limit, total_limit) if file_limit and total_limit else (
            file_limit or total_limit)
        data, normalization = import_root_file(filename, curr_limit)
        data.attrs = {} # Remove the acceptance Series, cannot be concatenated later and will be regenerated anyways
        data_list.append(data)
        total_normalization += normalization
        if total_limit:
            total_limit -= len(data_list[-1])
            if total_limit <= 0:
                break

    data = pd.concat(data_list)
    data.attrs["acceptances"] = pd.Series([len(data)/total_normalization if total_normalization!=0 else np.nan], index=["pre-selection"])
    return data, total_normalization


def clean_clusters(df: pd.DataFrame) -> None:
    """Clean the clusters in the Full dataframe. The variables are set to NaN for non-existing clusters

    :param df: Full dataframe
    """
    for cname in ["cluster1", "cluster2"]:
        df.loc[~df[f"{cname}_exists"], [f"{cname}_lkr_energy",
                                        f"{cname}_position_x", f"{cname}_position_y", f"{cname}_time"]] = np.nan


def clean_tracks(df: pd.DataFrame) -> None:
    """Clean the tracks in the Full dataframe. The variables are set to NaN (float variables),
    -99 (integer variables), False (boolean variables) for non-existing tracks.

    :param df: Full dataframe
    """
    for cname in ["track1", "track2", "track3"]:
        df.loc[~df[f"{cname}_exists"], [f"{cname}_rich_radius", f"{cname}_rich_center_x", f"{cname}_rich_center_y", f"{cname}_direction_x",
                                        f"{cname}_direction_y", f"{cname}_direction_z", f"{cname}_momentum_mag", f"{cname}_time", f"{cname}_lkr_energy"]] = np.nan
        df.loc[~df[f"{cname}_exists"], [
            f"{cname}_rich_hypothesis", f"{cname}_rich_nhits"]] = -99
        df.loc[~df[f"{cname}_exists"],  f"{cname}_has_muv3"] = False


def compute_derived(df: pd.DataFrame) -> None:
    """Compute some derived values from the Full dataframe (track E/p and Z position after magnet).

    :param df: Full dataframe
    """
    compute_eop(df, 1)
    compute_eop(df, 2)
    compute_eop(df, 3)

    df["track1_position_am_z"] = 180000
    df["track2_position_am_z"] = 180000
    df["track3_position_am_z"] = 180000


def compute_eop(df: pd.DataFrame, trackid: int) -> None:
    """Compute and set the E/p for a track in the Full dataframe.

    :param df: Full dataframe
    :param trackid: ID of the track to compute
    """
    df[f"track{trackid}_eop"] = track_eop(df, trackid)


def sample_normalization(fd: uproot.ReadOnlyDirectory, limit: Union[None, int] = None):
    """Return the normalization for the provided ROOT file. The normalization is the total number
    of events in the sample, corrected by the fraction of events extracted from the file.

    :param fd: ROOT file descriptor
    :param limit: Maximum number of events loaded in the file. If None, assume all the events were loaded. (default None)
    :return: Sample normalization
    """
    matrix = fd.get("export_flat/sel_matrix")
    tree = fd.get("export_flat/NA62Flat")
    total_events = matrix.values()[0][0]
    sel_events = tree.num_entries
    read_fraction = 1 if limit is None else limit/sel_events
    return total_events*read_fraction
