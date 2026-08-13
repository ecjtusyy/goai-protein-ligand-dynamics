import numpy as np

from src.baselines import linear_forecast, static_forecast
from src.metrics import rmsd_curve


def test_linear_forecast_recovers_constant_velocity():
    initial = np.array([[0.0, 1.0, 2.0], [2.0, 0.0, -1.0]])
    velocity = np.array([[0.5, -0.25, 1.0], [-1.0, 0.5, 0.25]])
    trajectory = np.stack([initial + t * velocity for t in range(7)])

    prediction = linear_forecast(trajectory[:3], horizon=4)

    np.testing.assert_allclose(prediction, trajectory[3:])
    np.testing.assert_allclose(rmsd_curve(prediction, trajectory[3:]), 0.0)


def test_static_forecast_repeats_last_frame():
    history = np.arange(18, dtype=float).reshape(3, 2, 3)

    prediction = static_forecast(history, horizon=4)

    assert prediction.shape == (4, 2, 3)
    np.testing.assert_array_equal(prediction, np.repeat(history[-1][None], 4, 0))
