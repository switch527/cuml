# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from cuml.ensemble.randomforestclassifier import RandomForestClassifier


class ExtraTreesClassifier(RandomForestClassifier):
    """
    Fits an ensemble of extremely randomized decision-tree classifiers.

    Each tree draws a single random threshold per candidate feature instead
    of scoring every quantile boundary, increasing bias and reducing variance
    relative to :class:`RandomForestClassifier`.

    .. note:: cuML draws the random threshold by sampling an integer bin
       index from the feature's global quantile grid (the same grid used by
       :class:`RandomForestClassifier`), then reading the corresponding
       quantile value. scikit-learn instead draws a continuous uniform on
       the node-local ``[min, max]`` feature range. At shallow nodes the
       two are equivalent; at deeper nodes cuML's draw may fall outside the
       node-local range and be rejected by ``min_samples_leaf``, in which
       case the candidate feature does not produce a split.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.ensemble import ExtraTreesClassifier as cuETC

        >>> X = cp.random.normal(size=(10, 4)).astype(cp.float32)
        >>> y = cp.asarray([0, 1] * 5, dtype=cp.int32)

        >>> model = cuETC(n_estimators=40, n_bins=8, max_depth=None)
        >>> model.fit(X, y)
        ExtraTreesClassifier()

    Parameters
    ----------
    bootstrap : boolean (default = False)
        Whether bootstrap samples are used when building trees.
        :class:`RandomForestClassifier` defaults to ``True``.
    max_depth : int or None (default = 16)
        Maximum tree depth. Use ``None`` for unlimited depth.

        .. rapids-pre-commit-hooks: disable-next-line
        .. versionchanged:: 26.08
          The default of `max_depth` will change from `16` to `None`.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Notes
    -----
    All other parameters and attributes are inherited from
    :class:`RandomForestClassifier`; see that class for the full list.
    ``sample_weight`` and ``class_weight`` (including ``'balanced'`` and
    ``'balanced_subsample'``) work through the same machinery. With the
    default ``bootstrap=False``, ``class_weight='balanced_subsample'``
    silently collapses to ``'balanced'`` (matching scikit-learn) since
    there is no per-tree subsample to weight by.

    For additional docs, see `scikit-learn's ExtraTreesClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html>`_.
    """

    _splitter = "random"
    _cpu_class_path = "sklearn.ensemble.ExtraTreesClassifier"

    def __init__(
        self,
        *,
        n_estimators=100,
        split_criterion="gini",
        bootstrap=False,
        max_samples=1.0,
        max_depth="deprecated",
        max_leaves=-1,
        max_features="sqrt",
        n_bins=128,
        min_samples_leaf=1,
        min_samples_split=2,
        min_impurity_decrease=0.0,
        max_batch_size=4096,
        random_state=None,
        n_streams=4,
        oob_score=False,
        class_weight=None,
        verbose=False,
        output_type=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            split_criterion=split_criterion,
            bootstrap=bootstrap,
            max_samples=max_samples,
            max_depth=max_depth,
            max_leaves=max_leaves,
            max_features=max_features,
            n_bins=n_bins,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            max_batch_size=max_batch_size,
            random_state=random_state,
            n_streams=n_streams,
            oob_score=oob_score,
            class_weight=class_weight,
            verbose=verbose,
            output_type=output_type,
        )
