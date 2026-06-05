#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#


import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score


@pytest.fixture(scope="module")
def classification_data():
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )
    return X, y


def test_etc_default_bootstrap_false(classification_data):
    X, y = classification_data
    clf = ExtraTreesClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.5


@pytest.mark.parametrize(
    "class_weight", [None, "balanced", "balanced_subsample", {0: 1, 1: 2}]
)
def test_etc_class_weight(classification_data, class_weight):
    X, y = classification_data
    clf = ExtraTreesClassifier(
        class_weight=class_weight,
        n_estimators=50,
        random_state=42,
    )
    clf.fit(X, y)
    assert clf.score(X, y) > 0.5


def test_etc_sample_weight_runs_on_gpu(classification_data):
    X, y = classification_data
    w = np.random.RandomState(0).uniform(0.5, 2.0, len(y)).astype(np.float32)
    clf = ExtraTreesClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y, sample_weight=w)
    y_pred = clf.predict(X)
    s_unweighted = clf.score(X, y)
    s_weighted = clf.score(X, y, sample_weight=w)
    assert s_unweighted == pytest.approx(accuracy_score(y, y_pred))
    assert s_weighted == pytest.approx(
        accuracy_score(y, y_pred, sample_weight=w)
    )
