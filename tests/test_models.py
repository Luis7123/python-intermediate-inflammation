"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
from inflammation.models import daily_mean, daily_max, daily_min
import pytest


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_max():
    """Test that mean function works for an array of zeros."""

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 5]])
    test_result = np.array([5, 5])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_min():
    """Test that mean function works for an array of zeros."""

    test_input = np.array([[1, 5],
                           [3, 4],
                           [5, 2]])
    test_result = np.array([1, 2])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)


@pytest.mark.parametrize(
    "test,expected",
    [
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]])
    ])
def test_patient_normalise(test, expected):
    """Test normalization works for arrays of one and positive integer
       assumption that test accuracy of two decimal places is sufficient"""
    from inflammation.models import patient_normalise
    npt.assert_almost_equal(patient_normalise(test), expected, decimal=2)
