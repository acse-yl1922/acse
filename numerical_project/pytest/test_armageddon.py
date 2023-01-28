
from collections import OrderedDict
import pandas as pd
import numpy as np
import os

from pytest import fixture, mark


# Use pytest fixtures to generate objects we know we'll reuse.
# This makes sure tests run quickly

@fixture(scope='module')
def armageddon():
    import armageddon
    return armageddon


@fixture(scope='module')
def planet(armageddon):
    return armageddon.Planet()


@fixture(scope='module')
def loc(armageddon):
    return armageddon.PostcodeLocator()


@fixture(scope='module')
def result(planet):
    input = {'radius': 1.,
             'velocity': 2.0e4,
             'density': 3000.,
             'strength': 1e5,
             'angle': 30.0,
             'init_altitude': 0.0,
             }

    result = planet.solve_atmospheric_entry(**input)

    return result


@fixture(scope='module')
def outcome(planet, result):
    outcome = planet.analyse_outcome(result=result)
    return outcome


def test_import(armageddon):
    assert armageddon


def test_planet_signature(armageddon):
    inputs = OrderedDict(atmos_func='exponential',
                         atmos_filename=None,
                         Cd=1., Ch=0.1, Q=1e7, Cl=1e-3,
                         alpha=0.3, Rp=6371e3,
                         g=9.81, H=8000., rho0=1.2)

    # call by keyword
    planet = armageddon.Planet(**inputs)

    # call by position
    planet = armageddon.Planet(*inputs.values())


def test_attributes(planet):
    for key in ('Cd', 'Ch', 'Q', 'Cl',
                'alpha', 'Rp', 'g', 'H', 'rho0'):
        assert hasattr(planet, key)


def test_atmos_filename(planet):

    assert os.path.isfile(planet.atmos_filename)


def test_solve_atmospheric_entry(result):

    assert type(result) is pd.DataFrame

    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time'):
        assert key in result.columns


def test_calculate_energy(planet, result):

    energy = planet.calculate_energy(result=result)

    assert type(energy) is pd.DataFrame

    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time', 'dedz'):
        assert key in energy.columns


def test_analyse_outcome(planet, outcome):

    assert type(outcome) is dict

    for key in ('outcome', 'burst_peak_dedz', 'burst_altitude',
                'burst_distance', 'burst_energy'):
        assert key in outcome.keys()


def test_damage_zones(armageddon):

    outcome = {'burst_peak_dedz': 1000.,
               'burst_altitude': 9000.,
               'burst_distance': 90000.,
               'burst_energy': 6000.,
               'outcome': 'Airburst'}

    blat, blon, damrad = armageddon.damage_zones(outcome, 55.0, 0.,
                                                 135., [27e3, 43e3])

    assert type(blat) is float
    assert type(blon) is float
    assert type(damrad) is list
    assert len(damrad) == 2


@mark.xfail
def test_great_circle_distance(armageddon):

    pnts1 = np.array([[54.0, 0.0], [55.0, 1.0], [54.2, -3.0]])
    pnts2 = np.array([[55.0, 1.0], [56.0, -2.1], [54.001, -0.003]])

    data = np.array([[1.28580537e+05, 2.59579735e+05, 2.25409117e+02],
                    [0.00000000e+00, 2.24656571e+05, 1.28581437e+05],
                    [2.72529953e+05, 2.08175028e+05, 1.96640630e+05]])

    dist = armageddon.great_circle_distance(pnts1, pnts2)

    assert np.allclose(data, dist, rtol=1.0e-4)


def test_locator_postcodes(loc):

    latlon = (52.2074, 0.1170)

    result = loc.get_postcodes_by_radius(latlon, [0.2e3, 0.1e3])

    assert type(result) is list
    if len(result) > 0:
        for element in result:
            assert type(element) is list


def test_locator_sectors(loc):

    latlon = (52.2074, 0.1170)

    result = loc.get_postcodes_by_radius(latlon, [3.0e3, 1.5e3], True)

    assert type(result) is list
    if len(result) > 0:
        for element in result:
            assert type(element) is list
