import pytest
import numpy as np
import pandas as pd
from .. import hlf

def get_unit_magnitude(vector):
    ''' Returns the magnitude of of the direction vector'''
    return np.sqrt(vector["direction_x"]**2 + vector["direction_y"]**2 + vector["direction_z"]**2)

class Test_ThreeVector:
    # Functions being tested
    sum_function = hlf.three_vectors_sum
    mag_function = hlf.three_vector_mag

    # Functions to generate some input randomly
    @staticmethod
    def generate_unit_vector(nvectors: int) -> pd.DataFrame:
        x = np.random.uniform(size=nvectors)
        y = np.random.uniform(size=nvectors)
        z = np.random.uniform(size=nvectors)
        mag = np.sqrt(x**2 + y**2 + z**2)
        x /= mag
        y /= mag
        z /= mag
        return pd.DataFrame({"direction_x": x, "direction_y": y, "direction_z": z})

    @staticmethod
    def generate_momentum(nvectors: int):
        return pd.DataFrame({"momentum_mag": np.random.uniform(0, 100000, nvectors)})

    @staticmethod
    def generate_vector():
        return Test_ThreeVector.generate_unit_vector(100).join(Test_ThreeVector.generate_momentum(100))

    # Fixtures generating known vectors
    @pytest.fixture
    def vector1(self):
        return pd.DataFrame({
            "direction_x": [0.540903, 0.650243, 0.870689, 0.652073, 0.726161, 0.075428, 0.609854, 0.514185, 0.839974, 0.153135],
            "direction_y": [0.565792, 0.391276, 0.345009, 0.400395, 0.218585, 0.313102, 0.428771, 0.823919, 0.370508, 0.513821],
            "direction_z": [0.622336, 0.651219, 0.350528, 0.643805, 0.651852, 0.946720, 0.666508, 0.238268, 0.396443, 0.844120],
            "momentum_mag": [46903.760347, 29950.543590, 772.051455, 50564.095810, 46457.921322, 35216.761041, 35296.008686, 30331.325132, 14299.457302, 20877.752034]})

    @pytest.fixture
    def vector2(self):
        return pd.DataFrame({
            "direction_x": [0.853424, 0.830252, 0.555158, 0.503505, 0.201706, 0.347256, 0.817377, 0.127174, 0.934739, 0.851988],
            "direction_y": [0.432551, 0.366020, 0.831525, 0.832755, 0.695402, 0.894849, 0.364707, 0.721017, 0.045276, 0.280187],
            "direction_z": [0.290804, 0.420370, 0.019098, 0.230223, 0.689733, 0.280462, 0.445964, 0.681147, 0.352438, 0.442281],
            "momentum_mag": [9746.595612, 2089.868306, 18351.161160, 10734.296881, 78923.692160, 23976.849705, 65345.755230, 2143.726276, 6434.717278, 51546.666629]})

    @pytest.fixture
    def vector_sum(self, vector1, vector2):
        return pd.DataFrame({
            "direction_x": [0.60445218, 0.66373376, 0.57286913, 0.64398297, 0.42157816, 0.20825984, 0.75285458, 0.49409809, 0.88044839, 0.70334033],
            "direction_y": [0.55179659, 0.39065836, 0.81899179, 0.48974082, 0.55218452, 0.61595781, 0.39146875, 0.82625722, 0.27300523, 0.37574975],
            "direction_z": [0.57459384, 0.63784288, 0.03276302, 0.58774132, 0.71928027, 0.75975247, 0.52911455, 0.2704923 , 0.38765807, 0.6034273],
            "momentum_mag": [ 55733.71152199,  31955.95888169,  18957.22794326,  59592.03196925, 117784.35646802,  52734.29043842,  99537.85438574,  32116.17093987, 20473.60615187,  66986.42759534]})

    def test_sum(self, vector1, vector2, vector_sum):
        ''' Test the sum function. The value provided by the sum_function need to be very close from the known sum '''
        sum = Test_ThreeVector.sum_function([vector1, vector2])
        assert(np.isclose(sum, vector_sum).all().all())

    def test_sum_is_unit(self, vector1, vector2):
        ''' Test the sum function. Make sure that the direction of the returned vector is unit '''
        sum = Test_ThreeVector.sum_function([vector1, vector2])
        assert(np.isclose(get_unit_magnitude(sum), 1).all())

    def test_magnitude(self, vector1):
        ''' Test the magnitude function. The value provided by the mag_function need to be very close from the known magnitude'''
        mag = Test_ThreeVector.mag_function(vector1)
        assert(np.isclose(mag, vector1["momentum_mag"]).all())

    def test_unit_direction(self, vector1):
        ''' Test that the given direction vector is unit '''
        mag = get_unit_magnitude(vector1)
        assert(np.isclose(mag, 1).all())

    def run_tests(self, sum_function, mag_function):
        ''' Run the tests manually, comparing the results on randomly generated vectors against the library functions '''
        Test_ThreeVector.sum_function = sum_function
        Test_ThreeVector.mag_function = mag_function

        v1 = Test_ThreeVector.generate_vector()
        v2 = Test_ThreeVector.generate_vector()
        vsum = hlf.three_vectors_sum([v1, v2])
        failed = False
        try:
            self.test_magnitude(v1)
        except AssertionError:
            print("[ERROR] Magnitude function does not return the expected values")
            failed = True
        try:
            self.test_sum(v1, v2, vsum)
        except AssertionError:
            print("[ERROR] Sum function does not return the expected values")
            failed = True
        try:
            self.test_sum_is_unit(v1, v2)
        except AssertionError:
            print("[ERROR] Sum function does not return a unit direction vector")
            failed = True

        if not failed:
            print("[INFO] All tests passed successfully")