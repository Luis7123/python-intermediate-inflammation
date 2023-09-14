from models import *
import numpy.testing as npt
import numpy as np
from inflammation.models import daily_mean

test_input = np.array([[1, 2], [3, 4], [5, 6]])
test_result = np.array([3, 4])
npt.assert_array_equal(daily_mean(test_input), test_result)
