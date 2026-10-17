"""Numerical helpers with NumPy pitfalls."""
import math

import numpy as np

readings = np.array([1.5, np.nan, 3.0], dtype=np.float64)

if readings[1] == np.nan:  # expect: ML.NUMPY.NAN_EQUALITY_CHECKER
    print("never true")
if np.nan != readings[0]:  # expect: ML.NUMPY.NAN_EQUALITY_CHECKER
    print("always true")
if readings[0] == float("nan"):  # expect: ML.NUMPY.NAN_EQUALITY_CHECKER
    print("never true either")
if readings[0] > np.nanmean(readings):  # expect: ML.NUMPY.MISSING_AXIS_SPECIFICATION
    print("fine: nanmean is not nan")
if math.isnan(readings[1]):
    print("fine")

noise = np.random.normal(0, 1, size=3)  # expect: ML.NUMPY.RANDOMNESS_CONTROL_CHECKER
rng = np.random.default_rng()  # expect: ML.NUMPY.RANDOMNESS_CONTROL_CHECKER
seeded = np.random.default_rng(42)

total = np.sum(readings)  # expect: ML.NUMPY.MISSING_AXIS_SPECIFICATION
column_means = np.mean(readings, axis=0)
peak = np.maximum(readings, 0)

batches = np.zeros(0, dtype=np.float64)
for chunk in range(3):
    batches = np.concatenate([batches, readings])  # expect: ML.NUMPY.INEFFICIENT_OPERATIONS

window = readings[0:2]  # expect: ML.NUMPY.COPY_VIEW_CONFUSION
window *= 2

picked = readings[[0, 2]]
picked[0] = 7.0

copy = readings[0:2].copy()
copy[0] = 1.0

values = [1, 2, 3]
part = values[1:3]
part[0] = 9
