import numpy as np
import pandas as pd
import uproot

from .hlf import track_eop


def import_root_file(filename: str) -> pd.DataFrame:
    fd = uproot.open(filename)
    x = fd.get("export_flat/NA62Flat")
    data = x.arrays(x.keys(), library="pd", entry_stop=1000000).rename(
        columns={"beam_momentum": "beam_momentum_mag", "beam_directionx": "beam_direction_x", "beam_directiony": "beam_direction_y", "beam_directionz": "beam_direction_z"})
    data = data.replace([np.inf, -np.inf], np.nan)
    type_dict = {"beam_momentum_mag": np.float64, "beam_direction_x": np.float64, "beam_direction_y": np.float64, "beam_direction_z": np.float64}
    for trackid in range(1,4):
        type_dict[f"track{trackid}_momentum_mag"] = np.float64
        type_dict[f"track{trackid}_direction_x"] = np.float64
        type_dict[f"track{trackid}_direction_y"] = np.float64
        type_dict[f"track{trackid}_direction_z"] = np.float64

    data = data.astype(type_dict)
    clean_clusters(data)
    clean_tracks(data)

    compute_derived(data)
    return data


def import_root_files(filenames: list[str]) -> pd.DataFrame:
    data_list = [import_root_file(_) for _ in filenames]
    return pd.concat(data_list)


def clean_clusters(df: pd.DataFrame) -> None:
    for cname in ["cluster1", "cluster2"]:
        df.loc[~df[f"{cname}_exists"], [f"{cname}_lkr_energy",
                                        f"{cname}_position_x", f"{cname}_position_y", f"{cname}_time"]] = np.nan


def clean_tracks(df: pd.DataFrame) -> None:
    for cname in ["track1", "track2", "track3"]:
        df.loc[~df[f"{cname}_exists"], [f"{cname}_rich_radius", f"{cname}_rich_center_x", f"{cname}_rich_center_y", f"{cname}_direction_x",
                                        f"{cname}_direction_y", f"{cname}_direction_z", f"{cname}_momentum_mag", f"{cname}_time", f"{cname}_lkr_energy"]] = np.nan
        df.loc[~df[f"{cname}_exists"], [
            f"{cname}_rich_hypothesis", f"{cname}_rich_nhits"]] = -99
        df.loc[~df[f"{cname}_exists"],  f"{cname}_has_muv3"] = False


def compute_derived(df: pd.DataFrame) -> None:
    compute_eop(df, 1)
    compute_eop(df, 2)
    compute_eop(df, 3)


def compute_eop(df: pd.DataFrame, trackid: int) -> None:
    df[f"track{trackid}_eop"] = track_eop(df, trackid)
