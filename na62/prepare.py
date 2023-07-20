import numpy as np
import pandas as pd
import uproot

from .hlf import compute_eop


def import_root_file(filename: str) -> pd.DataFrame:
    fd = uproot.open(filename)
    x = fd.get("export_flat/NA62Flat")
    data = x.arrays(x.keys(), library="pd", entry_stop=1000000).rename(
        columns={"beam_momentum": "beam_momentum_mag"})
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.astype({"beam_momentum_mag": np.float64, "track1_momentum_mag": np.float64,
                       "track2_momentum_mag": np.float64, "track3_momentum_mag": np.float64})
    clean_clusters(data)
    clean_tracks(data)
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
