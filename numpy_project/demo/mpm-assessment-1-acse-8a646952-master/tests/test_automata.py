"""Test automata module functions."""

import os
# from unittest import result

import numpy as np

import automata

BASE_PATH = os.path.dirname(__file__)


def test_lorenz96():
    """Test Lorenz 96 implementation"""
    initial64 = np.load(os.sep.join((BASE_PATH,
                                     'lorenz96_64_init.npy')))

    onestep64 = np.load(os.sep.join((BASE_PATH,
                                     'lorenz96_64_onestep.npy')))
    assert np.isclose(automata.lorenz96(initial64, 1), onestep64).all()

    thirtystep64 = np.load(os.sep.join((BASE_PATH,
                                        'lorenz96_64_thirtystep.npy')))
    assert np.isclose(automata.lorenz96(initial64, 30), thirtystep64).all()

# PYTHONPATH=$(echo $(pwd)) pytest tests/


def test_life():
    a = [[False, False, False],
         [True, True, True],
         [False, False, False]]
    result = [
        [False, False, False],
        [True, True, True],
        [False, False, False]

    ]
    assert np.isclose(automata.life(a, 4), result).all()


def test1_life_periodic():
    a = [[False, False, False, False],
         [True, False, False, False],
         [True, False, False, False],
         [True, False, False, False],
         [False, False, False, False]
         ]
    # result = [
    #     [False, False, False],
    #     [True, True, True],
    #     [False, False, False]

    # ]
    assert np.isclose(automata.life_periodic(a, 4), a).all()


def test_life2colour():
    a = [[0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0],
         [0, -1, 0, 1, 0],
         [0, 0, -1, -1, 0],
         [0, 0, 0, 0, 0]]
    result = [
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  1],
        [0,  0, -1,  0,  1],
        [0,  0,  0, -1, -1]

    ]

    assert np.isclose(automata.life2colour(a, 4), result).all()


def test_lifepent():
    a = [[0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 1, 0, 0],
         [0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]

         ]

    result = [[False, False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False, False],
              [False, False, False,  True,  True, False, False, False],
              [True,  True,  True,  True, False,  True, False, False],
              [False,  True,  True, False,  True,  True, False, False],
              [False, False, False, False, False, False, False, False],
              [False, False, False, False, False, False, False, False]
              ]

    assert np.isclose(automata.lifepent(a, 5), result).all()
    # PYTHONPATH=$(echo $(pwd)) pytest tests/
