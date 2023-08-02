from functools import reduce

import numpy as np
import pandas as pd
import pytest

from .. import constants, extract, hlf
from .test_vectors import Test_FourVector


class TestKinematics:
    inv_mass_fuction = hlf.invariant_mass
    total_momentum_function = hlf.total_momentum
    total_track_momentum_function = hlf.total_track_momentum
    missing_mass_sqr_function = hlf.missing_mass_sqr
    missing_mass_function = hlf.missing_mass
    propagation_function = hlf.propagate

    @staticmethod
    def generate_cluster(nclusters):
        return pd.DataFrame({"lkr_energy": np.random.uniform(low=5000, high=50000, size=nclusters),
                             "position_x": np.random.uniform(low=-1100, high=1100, size=nclusters),
                             "position_y": np.random.uniform(low=-1100, high=1100, size=nclusters)})

    @staticmethod
    def generate_vertex(nvertices):
        return pd.DataFrame({"vtx_x": np.random.uniform(low=-36, high=140, size=nvertices),
                            "vtx_y": np.random.uniform(low=-55, high=37, size=nvertices),
                             "vtx_z": np.random.uniform(low=-105000, high=176000, size=nvertices)})

    @staticmethod
    def generate_position(npositions):
        return pd.DataFrame({"position_x": np.random.uniform(low=-300, high=300, size=npositions),
                             "position_y": np.random.uniform(low=-300, high=300, size=npositions),
                             "position_z": [180000]*npositions})

    @staticmethod
    def generate_data_sample():
        t1 = hlf.set_mass(Test_FourVector.generate_vector(),
                          constants.pion_charged_mass).merge(TestKinematics.generate_position(100), left_index=True, right_index=True)
        t2 = hlf.set_mass(Test_FourVector.generate_vector(),
                          constants.pion_charged_mass).merge(TestKinematics.generate_position(100), left_index=True, right_index=True)
        t3 = hlf.set_mass(Test_FourVector.generate_vector(),
                          constants.pion_charged_mass).merge(TestKinematics.generate_position(100), left_index=True, right_index=True)
        c1 = TestKinematics.generate_cluster(100)
        c2 = TestKinematics.generate_cluster(100)
        beam = hlf.set_mass(Test_FourVector.generate_vector(),
                            constants.kaon_charged_mass)
        vertex = TestKinematics.generate_vertex(100)

        return reduce(lambda df1, df2: pd.merge(df1, df2, left_index=True, right_index=True),
                      [t1.add_prefix("track1_"), t2.add_prefix("track2_"), t3.add_prefix("track3_"),
                       c1.add_prefix("cluster1_"), c2.add_prefix("cluster2_"),
                       beam.add_prefix("beam_"), vertex])

    @pytest.fixture
    def track1(self):
        t = pd.DataFrame([[4.46814671e-03,  1.27984793e-03,  9.99989212e-01,
                           5.12514883e+04, 1.68548615e+02, -1.41565323e+02,  1.80000000e+05],
                          [1.17558450e-03,  2.01282115e-03,  9.99997258e-01,
                              2.90355469e+04, -8.17661057e+01, -3.15279144e+02,  1.80000000e+05],
                          [-4.58456017e-03,  1.79088046e-03,  9.99987900e-01,
                              3.38031484e+04, 1.75021561e+02, -3.56559052e+02,  1.80000000e+05],
                          [3.59912030e-03,  2.66916584e-03,  9.99989986e-01,
                              5.23151836e+04, -3.49859344e+02, -4.89549675e+01,  1.80000000e+05],
                          [8.48642271e-03, -1.34753075e-03,  9.99963105e-01,
                              2.52117676e+04, -5.93185463e+01,  4.37552368e+02,  1.80000000e+05],
                          [1.48418359e-03,  3.31244688e-03,  9.99993384e-01,
                              5.28407969e+04, -2.58427399e+02, -1.07184753e+02,  1.80000000e+05],
                          [-5.04365051e-03, -5.17357280e-03,  9.99973893e-01,
                              2.18096328e+04, -7.65993500e+01, -3.42173218e+02,  1.80000000e+05],
                          [-6.54260337e-04,  1.64348166e-04,  9.99999762e-01,
                              2.35864238e+04, 5.13477707e+00, -3.96109070e+02,  1.80000000e+05],
                          [2.17961357e-03,  3.69834783e-03,  9.99990761e-01,
                              4.92716914e+04, 4.57680237e+02,  2.37446640e+02,  1.80000000e+05],
                          [8.28949362e-03,  5.92988683e-03,  9.99948084e-01,
                              1.69399375e+04, 4.08505363e+01,  7.48688416e+02,  1.80000000e+05]],
                         columns=["direction_x", "direction_y", "direction_z", "momentum_mag", "position_x", "position_y", "position_z"])
        return hlf.set_mass(t, constants.pion_charged_mass).fillna(0)

    @pytest.fixture
    def track2(self):
        t = pd.DataFrame([[np.nan,             np.nan,             np.nan,
                           np.nan],
                          [4.50752355e-04,  3.21002956e-03,  9.99994755e-01,
                              1.66758398e+04],
                          [np.nan,             np.nan,             np.nan,
                              np.nan],
                          [np.nan,             np.nan,             np.nan,
                              np.nan],
                          [np.nan,             np.nan,             np.nan,
                              np.nan],
                          [np.nan,             np.nan,             np.nan,
                              np.nan],
                          [np.nan,             np.nan,             np.nan,
                              np.nan],
                          [3.82771890e-04, -1.87748775e-03,  9.99998152e-01,
                              3.53040703e+04],
                          [np.nan,             np.nan,             np.nan,
                              np.nan],
                          [np.nan,             np.nan,             np.nan,
                              np.nan]], columns=["direction_x", "direction_y", "direction_z", "momentum_mag"])
        return hlf.set_mass(t, constants.pion_charged_mass).fillna(0)

    @pytest.fixture
    def track3(self):
        t = pd.DataFrame([[np.nan,             np.nan,             np.nan,
                           np.nan],
                          [2.20005377e-03, -3.79692367e-03,  9.99990344e-01,
                              2.91333086e+04],
                          [np.nan,             np.nan,             np.nan,
                              np.nan],
                          [np.nan,             np.nan,             np.nan,
                              np.nan],
                          [np.nan,             np.nan,             np.nan,
                              np.nan],
                          [np.nan,             np.nan,             np.nan,
                              np.nan],
                          [np.nan,             np.nan,             np.nan,
                              np.nan],
                          [4.93847206e-03,  3.59693822e-03,  9.99981344e-01,
                              1.68171719e+04],
                          [np.nan,             np.nan,             np.nan,
                              np.nan],
                          [np.nan,             np.nan,             np.nan,
                              np.nan]], columns=["direction_x", "direction_y", "direction_z", "momentum_mag"])
        return hlf.set_mass(t, constants.pion_charged_mass).fillna(0)

    @pytest.fixture
    def cluster1(self):
        return pd.DataFrame([[4.92581348e+03,  4.41894379e+02, -3.57832825e+02],
                             [np.nan,             np.nan,             np.nan],
                             [1.00894033e+04,  1.03062146e+03,  2.25608093e+02],
                             [6.56341260e+03,  4.11558807e+02, -7.16979294e+01],
                             [1.99248438e+04,  2.89032669e+01,  3.10796112e+02],
                             [7.40126123e+03, -5.52373535e+02, -3.23934906e+02],
                             [1.97866777e+04,  3.11567749e+02,  4.93183105e+02],
                             [np.nan,             np.nan,             np.nan],
                             [5.03698877e+03, -4.35927094e+02,  2.35507523e+02],
                             [7.41148779e+03,  4.59994385e+02,  1.15334846e+02]], columns=["lkr_energy", "position_x", "position_y"])

    @pytest.fixture
    def cluster2(self):
        return pd.DataFrame([[1.80124785e+04, -6.41438110e+02, -1.58401398e+02],
                             [np.nan,             np.nan,             np.nan],
                             [3.16820645e+04,  4.86327850e+02, -2.60072662e+02],
                             [1.53680879e+04, -5.81759338e+02, -7.31715332e+02],
                             [3.02265781e+04, -2.17021591e+02, -6.34708939e+01],
                             [1.44756758e+04,  4.38117096e+02, -9.31059753e+02],
                             [3.06193711e+04,  4.53387421e+02,  2.34475441e+01],
                             [np.nan,             np.nan,             np.nan],
                             [1.91716562e+04,  3.74067383e+01, -9.31262756e+02],
                             [4.68691641e+04, -6.27283058e+01, -2.09504395e+02]], columns=["lkr_energy", "position_x", "position_y"])

    @pytest.fixture
    def beam(self):
        t = pd.DataFrame([[1.20999909e-03, 1.29999908e-05, 9.99999285e-01, 7.50370000e+04],
                          [1.20999909e-03, 1.29999908e-05,
                              9.99999285e-01, 7.50370000e+04],
                          [1.20999909e-03, 1.29999908e-05,
                              9.99999285e-01, 7.50370000e+04],
                          [1.20999909e-03, 1.29999908e-05,
                              9.99999285e-01, 7.50370000e+04],
                          [1.20999909e-03, 1.29999908e-05,
                              9.99999285e-01, 7.50370000e+04],
                          [1.20999909e-03, 1.29999908e-05,
                              9.99999285e-01, 7.50370000e+04],
                          [1.20999909e-03, 1.29999908e-05,
                              9.99999285e-01, 7.50370000e+04],
                          [1.20999909e-03, 1.29999908e-05,
                              9.99999285e-01, 7.50370000e+04],
                          [1.20999909e-03, 1.29999908e-05,
                              9.99999285e-01, 7.50370000e+04],
                          [1.20999909e-03, 1.29999908e-05, 9.99999285e-01, 7.50370000e+04]], columns=["direction_x", "direction_y", "direction_z", "momentum_mag"])
        return hlf.set_mass(t, constants.kaon_charged_mass)

    @pytest.fixture
    def vertex(self):
        return pd.DataFrame([[6.8722290e+01,  1.7496614e-01,  1.6062625e+05],
                             [1.5035622e+01, -9.6389418e+00,  1.2355704e+05],
                             [4.9778496e+01, -1.3737371e+00,  1.4542894e+05],
                             [6.6544937e+01, -4.8742042e+00,  1.5414425e+05],
                             [6.6157990e+01,  6.0045891e+00,  1.5769639e+05],
                             [5.8955772e+01,  2.0464820e-01,  1.5201898e+05],
                             [5.3736168e+01, -1.4992349e-01,  1.4805680e+05],
                             [4.5428818e+01,  1.4830046e+00,  1.3086898e+05],
                             [5.4024006e+01, -3.5741907e-01,  1.4680159e+05],
                             [6.0660263e+01, -1.4305631e+00,  1.5284966e+05]], columns=["vtx_x", "vtx_y", "vtx_z"])

    @pytest.fixture
    def data_sample(self, track1, track2, track3, cluster1, cluster2, beam, vertex):
        return reduce(lambda df1, df2: pd.merge(df1, df2, left_index=True, right_index=True),
                      [track1.add_prefix("track1_"), track2.add_prefix("track2_"), track3.add_prefix("track3_"),
                       cluster1.add_prefix(
                           "cluster1_"), cluster2.add_prefix("cluster2_"),
                       beam.add_prefix("beam_"), vertex])

    @pytest.fixture
    def total_momentum(self):
        return pd.Series([74188.44779569, 74844.32163879, 75573.26010186, 74245.20981644,
                          75362.01029751, 74716.2921539, 72214.51500157, 75707.32491176,
                          73478.90435421, 71219.53395486])

    @pytest.fixture
    def total_track_momentum(self):
        return pd.Series([51251.48893678, 74844.32163879, 33803.14887872, 52315.18492917,
                          25211.76815825, 52840.79536687, 21809.6327174, 75707.32491176,
                          49271.69020248, 16939.93792914])

    @pytest.fixture
    def invariant_mass(self):
        return pd.Series([139.32115883, 493.58132515, 139.46559405, 139.07719271, 139.46430308,
                          140.13789702, 139.58581464, 493.60073798, 139.99956115, 139.51902332]
                         )

    @pytest.fixture
    def missing_mass_sqr(self):
        return pd.Series([21188.95692229, -233.14927508, 16908.6418829, 15238.51949525,
                          19594.172297, 20324.43521106, 17165.12786102, -135.2683547,
                          19679.50644755, 13576.26178503])

    @pytest.fixture
    def missing_mass(self):
        return pd.Series([145.56427076, -15.26922641, 130.03323376, 123.4443984,  139.97918523,
                          142.56379348, 131.01575425, -11.63049245, 140.28366422, 116.51721669])

    @pytest.fixture
    def propagated(self):
        return pd.DataFrame([[4.41524047e+02, -6.33747299e+01,  2.41093000e+05],
                             [-9.94592491e+00, -1.92309524e+02,  2.41093000e+05],
                             [-1.05066363e+02, -2.47147468e+02,  2.41093000e+05],
                             [-1.29976086e+02,  1.14114014e+02,  2.41093000e+05],
                             [4.59161606e+02,  3.55224634e+02,  2.41093000e+05],
                             [-1.67753571e+02,  9.51839031e+01,  2.41093000e+05],
                             [-3.84739135e+02, -6.58250553e+02,  2.41093000e+05],
                             [-3.48359592e+01, -3.86068545e+02,  2.41093000e+05],
                             [5.90840599e+02,  4.63391891e+02,  2.41093000e+05],
                             [5.47306863e+02,  1.11098180e+03,  2.41093000e+05]], columns=["position_x", "position_y", "position_z"])

    def test_invariant_mass(self, track1, track2, track3, invariant_mass):
        inv_mass = TestKinematics.inv_mass_fuction([track1, track2, track3])
        assert (np.isclose(inv_mass, invariant_mass).all())

    def test_total_momentum(self, data_sample, total_momentum):
        assert (np.isclose(TestKinematics.total_momentum_function(
            data_sample), total_momentum).all())

    def test_total_track_momentum(self, data_sample, total_track_momentum):
        assert (np.isclose(TestKinematics.total_track_momentum_function(
            data_sample), total_track_momentum).all())

    def test_missing_mass_sqr(self, beam, track1, track2, track3, missing_mass_sqr):
        assert (np.isclose(TestKinematics.missing_mass_sqr_function(
            beam, [track1, track2, track3]), missing_mass_sqr).all())

    def test_missing_mass(self, beam, track1, track2, track3, missing_mass):
        assert (np.isclose(TestKinematics.missing_mass_function(
            beam, [track1, track2, track3]), missing_mass).all())

    def test_propagate(self, track1, propagated):
        assert (np.isclose(TestKinematics.propagation_function(
            track1, constants.lkr_position), propagated).all().all())

    def test_return_type_invariant_mass(self, track1, track2, track3):
        assert (type(TestKinematics.inv_mass_fuction(
            [track1, track2, track3])) == pd.Series)

    def test_return_type_total_momentum(self, data_sample):
        assert (type(TestKinematics.total_momentum_function(
            data_sample)) == pd.Series)

    def test_return_type_total_track_momentum(self, data_sample):
        assert (type(TestKinematics.total_track_momentum_function(
            data_sample)) == pd.Series)

    def test_return_type_missing_mass_sqr(self, beam, track1, track2, track3):
        assert (type(TestKinematics.missing_mass_sqr_function(
            beam, [track1, track2, track3])) == pd.Series)

    def test_return_type_missing_mass(self, beam, track1, track2, track3):
        assert (type(TestKinematics.missing_mass_function(
            beam, [track1, track2, track3])) == pd.Series)

    def test_return_type_propagate(self, track1):
        assert (type(TestKinematics.propagation_function(
            track1, constants.lkr_position)) == pd.DataFrame)

    def run_tests(self, *, inv_mass_fuction, total_momentum_function, total_track_momentum_function, missing_mass_sqr_function, missing_mass_function, propagation_function):
        ''' Run the tests manually, comparing the results against the library functions '''
        TestKinematics.inv_mass_fuction = inv_mass_fuction
        TestKinematics.total_momentum_function = total_momentum_function
        TestKinematics.total_track_momentum_function = total_track_momentum_function
        TestKinematics.missing_mass_sqr_function = missing_mass_sqr_function
        TestKinematics.missing_mass_function = missing_mass_function
        TestKinematics.propagation_function = propagation_function

        data = TestKinematics.generate_data_sample()
        t1 = extract.track(data, 1)
        t2 = extract.track(data, 2)
        t3 = extract.track(data, 3)
        beam = extract.get_beam(data)
        failed = False

        try:
            self.test_return_type_invariant_mass(t1, t2, t3)
        except (AssertionError, TypeError):
            print(
                "[ERROR] Invariant mass function does not return the expected data type (pandas.Series expected)")
            failed = True

        try:
            self.test_invariant_mass(
                t1, t2, t3, hlf.invariant_mass([t1, t2, t3]))
        except (AssertionError, TypeError):
            print("[ERROR] Invariant mass function does not return the expected values")
            failed = True

        try:
            self.test_return_type_total_momentum(data)
        except (AssertionError, TypeError):
            print(
                "[ERROR] Total momentum function does not return the expected data type (pandas.Series expected)")
            failed = True

        try:
            self.test_total_momentum(data, hlf.total_momentum(data))
        except (AssertionError, TypeError):
            print("[ERROR] Total momentum function does not return the expected values")
            failed = True

        try:
            self.test_return_type_total_track_momentum(data)
        except (AssertionError, TypeError):
            print(
                "[ERROR] Total track momentum function does not return the expected data type (pandas.Series expected)")
            failed = True

        try:
            self.test_total_track_momentum(
                data, hlf.total_track_momentum(data))
        except (AssertionError, TypeError):
            print(
                "[ERROR] Total track momentum function does not return the expected values")
            failed = True

        try:
            self.test_return_type_missing_mass_sqr(beam, t1, t2, t3)
        except (AssertionError, TypeError):
            print(
                "[ERROR] Missing mass squared function does not return the expected data type (pandas.Series expected)")
            failed = True

        try:
            self.test_missing_mass_sqr(
                beam, t1, t2, t3, hlf.missing_mass_sqr(beam, [t1, t2, t3]))
        except (AssertionError, TypeError):
            print(
                "[ERROR] Missing mass squared function does not return the expected values")
            failed = True

        try:
            self.test_return_type_missing_mass(beam, t1, t2, t3)
        except (AssertionError, TypeError):
            print(
                "[ERROR] Missing mass function does not return the expected data type (pandas.Series expected)")
            failed = True

        try:
            self.test_missing_mass(
                beam, t1, t2, t3, hlf.missing_mass(beam, [t1, t2, t3]))
        except (AssertionError, TypeError):
            print("[ERROR] Missing mass function does not return the expected values")
            failed = True

        try:
            self.test_return_type_propagate(t1)
        except (AssertionError, TypeError):
            print(
                "[ERROR] Propagate function does not return the expected data type (pandas.DataFrame expected)")
            failed = True

        try:
            self.test_propagate(t1, hlf.propagate(t1, constants.lkr_position))
        except (AssertionError, TypeError):
            print("[ERROR] Propagate function does not return the expected values")
            failed = True

        if not failed:
            print("[INFO] All tests passed successfully")
