import functools
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd

from . import constants
from .extract import cluster, get_beam, photon_momentum, track

################################################################
# Three-vector operations
################################################################

def three_vectors_sum(vectors: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Computes the 3-vector representing the sum of the input 3-vectors

    :param vectors: List of 3-vectors dataframes. Each dataframe must contain the variables 'direction_{x,y,z}' and 'momentum_mag'
    :return: 3-vector dataframe representing the sum of the input 3-vectors and containing the variables 'direction_{x,y,z}' and 'momentum_mag'
    """

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
    """
    Return the magnitude of a 3-vector

    :param vector: 3-vector dataframe. The dataframe must contain the variables 'momentum_mag'
    :return: Series representing the magnitude of the input vector
    """

    return vector["momentum_mag"]


def three_vector_invert(vector: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the inverse of a 3-vector (-vector)

    :param vector: 3-vector dataframe. The dataframe must contain the variables 'direction_{x,y,z}'
    :return: Copy of the input vector where the x,y,z coordinates have been inverted
    """

    neg_vector = vector.copy()
    neg_vector[["direction_x", "direction_y", "direction_z"]] *= -1

    return neg_vector


################################################################
# Four-vector operations
################################################################

def four_vectors_sum(vectors: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Computes the 4-vector representing the sum of the input 4-vectors

    :param vectors: List of 4-vectors dataframes. Each dataframe must contain the variables 'direction_{x,y,z}', 'momentum_mag' and 'energy'
    :return: 4-vector dataframe representing the sum of the input 4-vectors and containing the variables 'direction_{x,y,z}', 'momentum_mag' and 'energy'
    """

    if len(vectors) == 0:
        return pd.Series()

    momentum_sum = three_vectors_sum(vectors)

    p = vectors[0]
    momentum_sum["energy"] = p["energy"]
    for p in vectors[1:]:
        momentum_sum["energy"] += p["energy"]

    return momentum_sum


def four_vector_mag2(vector: pd.DataFrame) -> pd.DataFrame:
    """
    Return the magnitude squared of a 4-vector

    :param vector: 4-vector dataframe. The dataframe must contain the variables 'momentum_mag' and 'energy'
    :return: Series representing the magnitude squared of the input vector
    """

    return vector["energy"]**2 - three_vector_mag(vector)**2


def four_vector_mag(vector: pd.DataFrame) -> pd.DataFrame:
    """
    Return the magnitude of a 4-vector. The convention for vectors with negative magnitude squared is to
    take the square root of the absolute value of the magnitude squared and copy over the sign to the magnitude.

    :param vector: 4-vector dataframe. The dataframe must contain the variables 'momentum_mag' and 'energy'
    :return: Series representing the magnitude of the input vector
    """

    mag2 = four_vector_mag2(vector)
    return np.sign(mag2)*np.sqrt(np.abs(mag2))


def four_vector_invert(vector: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the inverse of a 4-vector (-vector)

    :param vector: 4-vector dataframe. The dataframe must contain the variables 'direction_{x,y,z}' and 'energy'
    :return: Copy of the input vector where the x,y,z,energy coordinates have been inverted
    """

    neg_vector = vector.copy()
    neg_vector[["direction_x", "direction_y", "direction_z", "energy"]] *= -1

    return neg_vector


################################################################
# Kinematic functions
################################################################
def _mass_assignment_to_objects(df: pd.DataFrame, mass_assignments: Dict[str, float]) -> List[pd.DataFrame]:
    track_masses = {}
    cluster_masses = {}
    for trackID in np.arange(1, 4):
        if f"track{trackID}" in mass_assignments:
            track_masses[trackID] = mass_assignments[f"track{trackID}"]
    for clusterID in np.arange(1, 3):
        if f"cluster{clusterID}" in mass_assignments:
            cluster_masses[clusterID] = mass_assignments[f"cluster{clusterID}"]

    # Extract the tracks and photons based on mass_assignments
    objects = []
    for trackID in track_masses:
        objects.append(set_mass(track(df, trackID), track_masses[trackID]))
    for clusterID in cluster_masses:
        objects.append(set_mass(photon_momentum(
            df, clusterID), cluster_masses[clusterID]))

    return objects


def invariant_mass(df_or_momenta: Union[pd.DataFrame, List[pd.DataFrame]], mass_assignments: Union[Dict[str, float], None] = None) -> pd.Series:
    """
    Compute the invariant mass. This function dispatches to :func:`invariant_mass_4vector` or
    :func:`invariant_mass_fulldf` based on the type of the 'mass_assignments' input.

    :param df_or_momenta: Full dataframe or list of 4-vector dataframes
    :return: Series representing the invariant mass of the input
    """

    if mass_assignments is not None:
        return invariant_mass_fulldf(df_or_momenta, mass_assignments)
    else:
        return invariant_mass_4vector(df_or_momenta)


def invariant_mass_4vector(momenta: List[pd.DataFrame]) -> pd.Series:
    """
    Compute the invariant mass of a list of momenta (4-vectors)

    :param momenta: List of 4-vector dataframes. The dataframe must contain the variables 'direction_{x,y,z}', 'momentum_mag' and 'energy'
    :return: Series representing the invariant mass of the input 4-vectors
    """

    total_four_momentum = four_vectors_sum(momenta)
    return four_vector_mag(total_four_momentum)


def invariant_mass_fulldf(df: pd.DataFrame, mass_assignments: Dict[str, float]) -> pd.Series:
    """Compute the invariant mass of a Full dataframe based on the specified mass assignment.

    :param df: Full dataframe
    :param mass_assignments: Dictionary of mass assignments. Each element of the dictionary should be a
        one of 'track{i}' where 'i' goes from 1 to 3, or 'cluster{j}' where 'j' goes from 1 to 2, and the
        value should be the mass to associate to the object. Omit objects that should not be included in
        the computation
    :return: Series representing the invariant mass
    """
    objects = _mass_assignment_to_objects(df, mass_assignments)
    return invariant_mass_4vector(objects)


def total_momentum(df: pd.DataFrame) -> pd.Series:
    """
    Compute the total momentum over all existing tracks and clusters in each event.

    :param df: Full dataframe
    :return: Series representing the total momentum magnitude
    """

    t1 = track(df, 1).fillna(0)
    t2 = track(df, 2).fillna(0)
    t3 = track(df, 3).fillna(0)
    c1 = photon_momentum(df, 1).fillna(0)
    c2 = photon_momentum(df, 2).fillna(0)
    return three_vector_mag(three_vectors_sum([t1, t2, t3, c1, c2]))


def total_track_momentum(df: pd.DataFrame) -> pd.Series:
    """
    Compute the total momentum over all existing tracks in each event.

    :param df: Full dataframe
    :return: Series representing the total momentum magnitude
    """

    t1 = track(df, 1).fillna(0)
    t2 = track(df, 2).fillna(0)
    t3 = track(df, 3).fillna(0)
    return three_vector_mag(three_vectors_sum([t1, t2, t3]))


def missing_mass_sqr(df: pd.DataFrame, momenta_or_masses: Union[Dict[str, float], List[pd.DataFrame]]) -> pd.Series:
    """
    Compute the missing mass squared. This function dispatches to :func:`missing_mass_sqr_from_4vector` or
    :func:`missing_mass_sqr_from_fulldf` based on the type of the 'momenta_or_masses' input.

    :param df: Beam momentum 4-vector dataframe or full dataframe
    :param momenta_or_masses: List of 4-vector dataframes or dictionary of mass assignments
    :return: Series representing the missing mass squared
    """

    if isinstance(momenta_or_masses, dict):
        return missing_mass_sqr_from_fulldf(df, momenta_or_masses)
    else:
        return missing_mass_sqr_from_4vector(df, momenta_or_masses)


def missing_mass_sqr_from_4vector(beam: pd.DataFrame, momenta: List[pd.DataFrame]) -> pd.Series:
    """
    Compute the missing mass squared.

    :param beam: 4-vector dataframe
    :param momenta: List of 4-vector dataframes
    :return: Series representing the missing mass squared
    """

    momenta_sum = four_vectors_sum(momenta)
    return four_vector_mag2(four_vectors_sum([beam, four_vector_invert(momenta_sum)]))


def missing_mass_sqr_from_fulldf(df: pd.DataFrame, mass_assignments: Dict[str, float]) -> pd.Series:
    """
    Compute the missing mass squared.

    :param df: Full dataframe
    :param mass_assignments: Dictionary of mass assignments. Each element of the dictionary should be a
        one of 'track{i}' where 'i' goes from 1 to 3, or 'cluster{j}' where 'j' goes from 1 to 2, and the
        value should be the mass to associate to the object. Omit objects that should not be included in
        the computation
    :return: Series representing the missing mass squared
    """

    # Extract the beam
    beam = set_mass(get_beam(df), constants.kaon_charged_mass)

    objects = _mass_assignment_to_objects(df, mass_assignments)
    return missing_mass_sqr_from_4vector(beam, objects)


def missing_mass(beam: pd.DataFrame, momenta: List[pd.DataFrame]) -> pd.Series:
    """
    Compute the missing mass. The convention for negative missing mass squared is to
    take the square root of the absolute value of the missing mass squared and copy over
    the sign to the missing mass.

    :param beam: 4-vector dataframe
    :param momenta: List of 4-vector dataframes
    :return: Series representing the missing mass
    """

    momenta_sum = four_vectors_sum(momenta)
    return four_vector_mag(four_vectors_sum([beam, four_vector_invert(momenta_sum)]))


def propagate(track: pd.DataFrame, z_final: int, position_field_name: str = "position", direction_field_name: str = "direction") -> pd.DataFrame:
    """
    Compute the position vector of a track at a given Z position.

    :param track: track dataframe. Must contain three direction ('{direction_field_name}_{x,y,z}') and three
        position components ('{position_field_name}_{x,y,z}')
    :param z_final: Z position where the track should be propagated
    :param position_field_name: Name of the position field in the track dataframe (default to 'position')
    :param direction_field_name: Name of the position field in the track dataframe (default to 'direction')
    :return: New position dataframe containing the variables 'position_{x,y,z}'
    """

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
    """
    Invert a callable cut

    :param fun: Callable to invert
    :return: New callable cut with inverted logic with respect to input
    """

    def not_cut(df: pd.DataFrame) -> pd.Series:
        return ~fun(df)
    return not_cut


def combine_cuts(cuts: List[Callable], booleans: Union[None, List[pd.Series], List[np.array]] = None) -> Callable:
    """
    Combine a list of cuts into a single Callable

    :param cuts: List of callables to combine
    :return: New callable combining all the input callables
    """

    def cut(df: pd.DataFrame) -> pd.Series:
        all_cuts = map(lambda cut: cut(df), cuts)
        return functools.reduce(lambda c1, c2: c1 & c2, all_cuts)
    return cut


def select(df: pd.DataFrame, cuts: List[Callable]) -> pd.DataFrame:
    """
    Apply a list of cuts to a dataframe

    :param df: Dataframe
    :param cuts: List of callables
    :return: DataFrame to which the cuts have been applied
    """

    return df.loc[combine_cuts(cuts)]


def make_min_max_cut(min_val: Union[None, int], max_val: Union[None, int], *,
                     which_value: Union[str, None] = None, which_object: Union[None, str] = None,
                     df_transform: Union[None, Callable] = None, **kwargs
                     ) -> Callable:
    """Create a cut on a value. The cut can be applied to a dataframe, or to a series directly.

    Use to implement other cuts. Requires either which_value to be set, or df_transform (exclusive).

    :param min_val: Minimum value that should be kept. If 'None', no minimum is applied
    :param max_val: Maximum value that should be kept. If 'None', no maximum is applied
    :param which_value: Name of the variable on which the cut should be applied when a dataframe is passed (exclusive with `df_transform`)
    :param which_object: If the variable is prefixed by an object name, specify here which object (e.g. 'track1_{which_value}').
        If not, `None` will apply to the variable itself.
    :param df_transform: Transformation function to apply to the input dataframe to transform it into the series on which the cut is applied
        (exclusive with `which_value`). Any parameter to be passed to the function can be passed as `**kwargs`.
    :return: Callable computing the alignable boolean Series representing the cut
    """

    if df_transform and which_value:
        raise ValueError("Cannot specify both df_transform and which_value. They are exclusive.")
    if not df_transform and not which_value:
        raise ValueError("Must specify either of df_transform and which_value.")

    if which_object:
        which_object = _select_object(which_object)
        transform = f"{which_object}{which_value}"
    elif df_transform:
        transform = functools.partial(df_transform, **kwargs)

    def cut(df: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        serie_cut = _select_series(df, transform)
        min_range = serie_cut > min_val if min_val else True
        max_range = serie_cut < max_val if max_val else True
        return min_range & max_range
    return cut


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
    """
    Create a E/p cut. The cut can be applied either to a full dataframe or to a track dataframe.

    :param min_eop: Minimum E/p value that should be kept. If 'None', no minimum is applied
    :param max_eop: Maximum E/p value that should be kept. If 'None', no maximum is applied
    :param which_track: Do not specify (None) if the dataframe to which the cut applies is a track dataframe.
        If applied to a full dataframe, the name of the track should be specified (e.g. 'track1')
    :return: Callable computing the alignable boolean Series representing the cut
    """

    return make_min_max_cut(min_eop, max_eop, which_value="eop", which_object=which_track)


def make_rich_cut(rich_hypothesis: Union[str, int],
                  min_p: Union[None, int] = 15000, max_p: Union[None, int] = 40000,
                  which_track: Union[None, str] = None) -> Callable:
    """
    Create a cut on a track RICH hypothesis, including a cut on the track
    momentum (default can be changed or disabled with 'min_p' and 'max_p' parameters).
    The cut can be applied either to a full dataframe or to a track dataframe.

    :param rich_hypothesis: RICH hypothesis to choose. Specify as a string from one of the values present in :data:`na62.constants.rich_hypothesis_map`
    :param min_p: Minimum momentum value that should be kept. If 'None', no minimum is applied
    :param max_p: Maximum momentum value that should be kept. If 'None', no maximum is applied
    :param which_track: Do not specify (None) if the dataframe to which the cut applies is a track dataframe.
        If applied to a full dataframe, the name of the track should be specified (e.g. 'track1')
    :return: Callable computing the alignable boolean Series representing the cut
    """

    if isinstance(rich_hypothesis, str):
        rich_hypothesis = constants.rich_hypothesis_map[rich_hypothesis]
    momentum_condition = make_momentum_cut(min_p, max_p)
    which_track = _select_object(which_track)

    def cut(df: pd.DataFrame) -> pd.Series:
        hypothesis = df[f"{which_track}rich_hypothesis"] == rich_hypothesis
        return hypothesis & momentum_condition(df)
    return cut


def make_muv3_cut(has_muv3: bool, which_track: Union[None, str] = None) -> Callable:
    """
    Create a cut on the presence of a track MUV3 signal. The cut can be applied either to a full dataframe or to a track dataframe.

    :param has_muv3: True if the track must have a MUV3 signal, else False
    :param which_track: Do not specify (None) if the dataframe to which the cut applies is a track dataframe.
        If applied to a full dataframe, the name of the track should be specified (e.g. 'track1')
    :return: Callable computing the alignable boolean Series representing the cut
    """

    which_track = _select_object(which_track)

    def cut(df: pd.DataFrame) -> pd.Series:
        return df[f"{which_track}has_muv3"] == has_muv3
    return cut


def make_momentum_cut(min_p: Union[None, int], max_p: Union[None, int], which_object: Union[None, str] = None) -> Callable:
    """
    Create a momentum cut. The cut can be applied either to a full dataframe or to a track dataframe.

    :param min_p: Minimum momentum value that should be kept. If 'None', no minimum is applied
    :param max_p: Maximum momentum value that should be kept. If 'None', no maximum is applied
    :param which_object: Do not specify (None) if the dataframe to which the cut applies is a 3-vector or 4-vector dataframe.
        If applied to a full dataframe, the name of the object (track or cluster) should be specified (e.g. 'track1')
    :return: Callable computing the alignable boolean Series representing the cut
    """

    return make_min_max_cut(min_p, max_p, which_value="momentum_mag", which_object=which_object)


def make_total_momentum_cut(min_p: Union[None, int], max_p: Union[None, int] = None) -> Callable:
    """
    Create a total momentum cut. The cut can be applied to a full dataframe.

    :param min_p: Minimum momentum value that should be kept. If 'None', no minimum is applied
    :param max_p: Maximum momentum value that should be kept. If 'None', no maximum is applied
    :return: Callable computing the alignable boolean Series representing the cut
    """

    return make_min_max_cut(min_p, max_p, df_transform=total_momentum)


def make_missing_mass_sqr_cut(min_mm2: Union[None, float], max_mm2: Union[None, float], mass_assignments: Dict[str, float]) -> Callable:
    """
    Create a missing mass squared cut. The cut can be applied to a full dataframe.

    :param min_mm2: Minimum missing mass squared value that should be kept. If 'None', no minimum is applied
    :param max_mm2: Maximum missing mass squared value that should be kept. If 'None', no maximum is applied
    :param mass_assignments: Dictionary of mass assignments. See :func:`missing_mass_sqr_from_fulldf`.
    :return: Callable computing the alignable boolean Series representing the cut
    """

    return make_min_max_cut(min_mm2, max_mm2, df_transform=missing_mass_sqr, momenta_or_masses=mass_assignments)


def make_invariant_mass_cut(min_mass: Union[None, float], max_mass: Union[None, float], mass_assignments: Dict[str, float]) -> Callable:

    return make_min_max_cut(min_mass, max_mass, df_transform=invariant_mass, mass_assignments=mass_assignments)


def make_exists_cut(exists: Union[None, List[str]], not_exists: Union[None, List[str]]) -> Callable:
    """
    Create a cut to select only events where some objects (tracks or clusters) exist. The cut can be applied to a full dataframe.

    :param exists: List of objects that should exist. 'None' or empty list to disable.
    :param not_exists: List of objects that should not exist. 'None' or empty list to disable.
    :return: Callable computing the alignable boolean Series representing the cut
    """

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
    """
    Compute the total LKr energy over all existing tracks and clusters in each event.

    :param df: Full dataframe
    :return: Series representing the total LKr energy
    """

    t1 = track(df, 1)
    t2 = track(df, 2)
    t3 = track(df, 3)
    c1 = cluster(df, 1)
    c2 = cluster(df, 2)
    return t1["lkr_energy"].fillna(0) + t2["lkr_energy"].fillna(0) + t3["lkr_energy"].fillna(0) + c1["lkr_energy"].fillna(0) + c2["lkr_energy"].fillna(0)


def track_eop(df: pd.DataFrame, trackid: int) -> pd.Series:
    """
    Compute the E/p for the specified track

    :param df: Full dataframe
    :param trackid: Track to compute
    :return: Series representing the track E/p
    """

    t = track(df, trackid)
    return t["lkr_energy"]/t["momentum_mag"]


def set_mass(momentum: pd.DataFrame, mass: float) -> pd.DataFrame:
    """
    Update a momentum by assigning a mass (in-place). The 'energy' of the vector is updated.
    The 'mass' variable is also set for completeness (redundant)

    :param momentum: 3-vector or 4-vector dataframe to update
    :return: Updated input momentum with mass assignment.
    """

    momentum["mass"] = mass
    momentum["energy"] = np.sqrt(mass**2 + momentum["momentum_mag"]**2)
    return momentum


def ring_radius(p: Union[float, np.array, pd.Series], mass: float) -> Union[float, np.array, pd.Series]:
    """
    Compute the expected theoretical NA62 RICH ring radius based on the momentum and mass.

    :param p: Momentum of the particle.
        Can be a numpy array or a pandas.Series to perform the computation directly on a range of momenta.
    :param mass: Mass of the particle.
    :return: Radius as a function of the momenta passed in input (same format as the input param 'p')
    """

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

def _select_object(which_object: Union[None, str]) -> str:
    if which_object is None:
        return ""
    else:
        return f"{which_object}_"

def _select_series(df: Union[pd.DataFrame, pd.Series], transform: Union[Callable, str]) -> pd.Series:
    if isinstance(df, pd.DataFrame):
        if isinstance(transform, str):
            return df[transform]
        else:
            return transform(df)
    else:
        return df
