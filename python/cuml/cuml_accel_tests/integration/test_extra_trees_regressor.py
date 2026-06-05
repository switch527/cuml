#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#


import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score


@pytest.fixture(scope="module")
def regression_data():
    X, y = make_regression(
        n_samples=300,
        n_features=20,
        n_informative=10,
        noise=0.1,
        random_state=42,
    )
    return X, y


def test_etr_default_bootstrap_false(regression_data):
    X, y = regression_data
    reg = ExtraTreesRegressor(n_estimators=50, random_state=42)
    reg.fit(X, y)
    assert reg.score(X, y) > 0.5


def test_etr_sample_weight_runs_on_gpu(regression_data):
    X, y = regression_data
    w = np.random.RandomState(0).uniform(0.5, 2.0, len(y)).astype(np.float32)
    reg = ExtraTreesRegressor(n_estimators=10, random_state=42)
    reg.fit(X, y, sample_weight=w)
    y_pred = reg.predict(X)
    s_unweighted = reg.score(X, y)
    s_weighted = reg.score(X, y, sample_weight=w)
    assert s_unweighted == pytest.approx(r2_score(y, y_pred))
    assert s_weighted == pytest.approx(r2_score(y, y_pred, sample_weight=w))
