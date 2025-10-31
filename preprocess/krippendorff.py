"""
This module provides a function to compute the Krippendorff's alpha statistical measure of the agreement achieved
when coding a set of units based on the values of a variable, plus a new function to compute per-item alpha.
For more information, see: https://en.wikipedia.org/wiki/Krippendorff%27s_alpha
"""

from __future__ import annotations

from typing import Literal, Protocol, TypeVar, Union

import numpy as np
import numpy.typing as npt

DEFAULT_DTYPE = np.float_

ValueScalarType = TypeVar("ValueScalarType", bound=np.generic)
MetricResultScalarType = TypeVar("MetricResultScalarType", bound=np.inexact)


class DistanceMetric(Protocol):
    def __call__(self,
                 v1: npt.NDArray[ValueScalarType],
                 v2: npt.NDArray[ValueScalarType],
                 i1: npt.NDArray[np.int_],
                 i2: npt.NDArray[np.int_],
                 n_v: npt.NDArray[MetricResultScalarType],
                 dtype: np.dtype[MetricResultScalarType] = DEFAULT_DTYPE
                 ) -> npt.NDArray[MetricResultScalarType]:
        """Computes the distance for two arrays element-wise.

        Parameters
        ----------
        v1 : ndarray
            First array of values.

        v2 : ndarray
            Second array of values.

        i1 : ndarray
            Ordinal indices of the first array (with the same shape).

        i2 : ndarray
            Ordinal indices of the second array (with the same shape).

        n_v : ndarray, shape (V,)
            Number of pairable elements for each value.

        dtype : data-type
            Result and computation data-type.

        Returns
        -------
        d : ndarray
            Element-wise distance between v1 and v2.
        """


LevelOfMeasurement = Union[Literal["nominal", "ordinal", "interval", "ratio"], DistanceMetric]


def _nominal_metric(v1, v2, i1, i2, n_v, dtype=DEFAULT_DTYPE):
    return (v1 != v2).astype(dtype)


def _ordinal_metric(v1, v2, i1, i2, n_v, dtype=DEFAULT_DTYPE):
    i1, i2 = np.minimum(i1, i2), np.maximum(i1, i2)
    ranges = np.dstack((i1, i2 + 1))
    sums_between_indices = np.add.reduceat(np.append(n_v, 0),
                                          ranges.reshape(-1))[::2].reshape(*i1.shape)
    return (sums_between_indices - np.divide(n_v[i1] + n_v[i2], 2, dtype=dtype)) ** 2


def _interval_metric(v1, v2, i1, i2, n_v, dtype=DEFAULT_DTYPE):
    return (v1 - v2).astype(dtype) ** 2


def _ratio_metric(v1, v2, i1, i2, n_v, dtype=DEFAULT_DTYPE):
    v1_plus_v2 = v1 + v2
    return np.divide(v1 - v2,
                     v1_plus_v2,
                     out=np.zeros(np.broadcast(v1, v2).shape),
                     where=v1_plus_v2 != 0,
                     dtype=dtype) ** 2


def _coincidences(value_counts: npt.NDArray[np.int_],
                  dtype: np.dtype[MetricResultScalarType] = DEFAULT_DTYPE
                  ) -> npt.NDArray[MetricResultScalarType]:
    """Aggregate coincidence matrix from all items."""
    N, V = value_counts.shape
    pairable = np.maximum(value_counts.sum(axis=1), 2)
    diagonals = value_counts[:, np.newaxis, :] * np.eye(V)[np.newaxis, ...]
    unnormalized_coincidences = (value_counts[..., np.newaxis]
                                 * value_counts[:, np.newaxis, :]
                                 - diagonals)
    return np.divide(unnormalized_coincidences,
                     (pairable - 1).reshape((-1, 1, 1)),
                     dtype=dtype).sum(axis=0)


def _random_coincidences(n_v: npt.NDArray[MetricResultScalarType],
                         dtype: np.dtype[MetricResultScalarType] = DEFAULT_DTYPE
                         ) -> npt.NDArray[MetricResultScalarType]:
    """Random (chance) coincidence matrix from marginals."""
    return np.divide(np.outer(n_v, n_v) - np.diagflat(n_v),
                     n_v.sum() - 1,
                     dtype=dtype)


def _distances(value_domain: npt.NDArray[ValueScalarType],
               distance_metric: DistanceMetric,
               n_v: npt.NDArray[np.int_],
               dtype: np.dtype[MetricResultScalarType] = DEFAULT_DTYPE
               ) -> npt.NDArray[MetricResultScalarType]:
    """Distance matrix for all pairs in the domain."""
    indices = np.arange(len(value_domain))
    return distance_metric(value_domain[:, np.newaxis],
                           value_domain[np.newaxis, :],
                           i1=indices[:, np.newaxis],
                           i2=indices[np.newaxis, :],
                           n_v=n_v,
                           dtype=dtype)


def _distance_metric(level_of_measurement: LevelOfMeasurement) -> DistanceMetric:
    if level_of_measurement == "nominal":
        return _nominal_metric
    elif level_of_measurement == "ordinal":
        return _ordinal_metric
    elif level_of_measurement == "interval":
        return _interval_metric
    elif level_of_measurement == "ratio":
        return _ratio_metric
    else:
        # Custom callable
        return level_of_measurement


def _reliability_data_to_value_counts(reliability_data: npt.NDArray[ValueScalarType],
                                      value_domain: npt.NDArray[ValueScalarType]
                                      ) -> npt.NDArray[np.int_]:
    """Map each item to how many coders assigned each value."""
    return (reliability_data.T[..., np.newaxis]
            == value_domain[np.newaxis, np.newaxis, :]).sum(axis=1)


def alpha(reliability_data: npt.ArrayLike | None = None,
          value_counts: npt.ArrayLike | None = None,
          value_domain: npt.ArrayLike | None = None,
          level_of_measurement: LevelOfMeasurement = "interval",
          dtype: npt.DTypeLike = DEFAULT_DTYPE) -> float:
    """
    Compute Krippendorff's alpha for the entire dataset.

    See https://en.wikipedia.org/wiki/Krippendorff%27s_alpha for more information.

    Parameters
    ----------
    reliability_data : array_like, shape (M, N)
        Each row is a coder; each column is an item.

    value_counts : array_like, shape (N, V)
        For each item, how many coders assigned each possible value.
        If provided, `reliability_data` must not be provided.

    value_domain : array_like, shape (V,)
        The possible values the units can take.
        Must be sorted or explicitly match the measurement scale if not nominal.

    level_of_measurement : {"nominal", "ordinal", "interval", "ratio"} or callable
        Distance metric or scale of measurement.

    dtype : inexact data-type
        Floating data type for the result.

    Returns
    -------
    alpha : float
        Krippendorff's alpha for the entire dataset.
    """
    if (reliability_data is None) == (value_counts is None):
        raise ValueError("Either reliability_data or value_counts must be provided, but not both.")

    if value_counts is None:
        reliability_data = np.asarray(reliability_data)
        kind = reliability_data.dtype.kind

        if kind in {"i", "u", "f"}:
            # Numeric data
            computed_value_domain = np.unique(reliability_data[~np.isnan(reliability_data)])
        elif kind in {"U", "S"}:
            # String data
            computed_value_domain = np.unique(reliability_data[reliability_data != "nan"])
        else:
            raise ValueError(f"Cannot construct domain for dtype kind {kind}.")

        if value_domain is None:
            if kind in {"U", "S"} and level_of_measurement != "nominal":
                raise ValueError("When using strings with a non-nominal scale, "
                                 "an ordered value_domain is required.")
            value_domain = computed_value_domain
        else:
            value_domain = np.asarray(value_domain)
            if not np.isin(computed_value_domain, value_domain).all():
                raise ValueError("Data contains out-of-domain values.")

        value_counts = _reliability_data_to_value_counts(reliability_data, value_domain)
    else:
        value_counts = np.asarray(value_counts)
        if value_domain is None:
            value_domain = np.arange(value_counts.shape[1])
        else:
            value_domain = np.asarray(value_domain)
            if value_counts.shape[1] != len(value_domain):
                raise ValueError("value_domain size must match number of columns in value_counts.")

    if len(value_domain) <= 1:
        raise ValueError("There must be at least two distinct values in the domain.")

    if (value_counts.sum(axis=-1) <= 1).all():
        raise ValueError("At least one item must be coded by two or more coders.")

    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.inexact):
        raise ValueError("`dtype` must be an inexact floating type.")

    distance_metric = _distance_metric(level_of_measurement)

    # Observed coincidence matrix
    o = _coincidences(value_counts, dtype=dtype)
    # Category marginals
    n_v = o.sum(axis=0)
    # "Random" coincidence matrix
    e = _random_coincidences(n_v, dtype=dtype)
    # Distance matrix
    d = _distances(value_domain, distance_metric, n_v, dtype=dtype)

    # Krippendorff's alpha
    return 1 - (o * d).sum() / (e * d).sum()


def alpha_per_item(reliability_data: npt.ArrayLike | None = None,
                   value_counts: npt.ArrayLike | None = None,
                   value_domain: npt.ArrayLike | None = None,
                   level_of_measurement: LevelOfMeasurement = "interval",
                   dtype: npt.DTypeLike = DEFAULT_DTYPE
                   ) -> np.ndarray:
    """
    Compute a Krippendorff's alpha value per item (mirroring the Java method).

    This uses the same global "expected disagreement" as the standard alpha computation,
    but calculates each item's "observed disagreement" separately. The formula is:

        alpha_i = 1 - (D_O_i / D_E)

    where:
    - D_O_i is the per-item observed disagreement,
    - D_E is the single global expected disagreement used for all items.

    Parameters
    ----------
    reliability_data : array_like, shape (M, N)
        Each row is a coder; each column is an item.

    value_counts : array_like, shape (N, V)
        Number of coders that assigned each possible value for each item.

    value_domain : array_like, shape (V,)
        The set of possible values.

    level_of_measurement : {"nominal", "ordinal", "interval", "ratio"} or callable
        Distance metric or scale of measurement.

    dtype : inexact data-type
        Floating data type for intermediate and final results.

    Returns
    -------
    per_item_alphas : ndarray, shape (N,)
        Krippendorff's alpha for each item.
    """
    # First, reuse the same setup as `alpha` to get:
    # - validated `value_counts` and `value_domain`
    # - global distance matrix `d`
    # - global random matrix `e` => used to get D_E
    # - do not overwrite `value_counts`, so we do a local copy if needed
    if (reliability_data is None) == (value_counts is None):
        raise ValueError("Either reliability_data or value_counts must be provided, but not both.")

    if value_counts is None:
        reliability_data = np.asarray(reliability_data)
        kind = reliability_data.dtype.kind

        if kind in {"i", "u", "f"}:
            computed_value_domain = np.unique(reliability_data[~np.isnan(reliability_data)])
        elif kind in {"U", "S"}:
            computed_value_domain = np.unique(reliability_data[reliability_data != "nan"])
        else:
            raise ValueError(f"Cannot construct domain for dtype kind {kind}.")

        if value_domain is None:
            if kind in {"U", "S"} and level_of_measurement != "nominal":
                raise ValueError("When using strings with a non-nominal scale, "
                                 "an ordered value_domain is required.")
            value_domain = computed_value_domain
        else:
            value_domain = np.asarray(value_domain)
            if not np.isin(computed_value_domain, value_domain).all():
                raise ValueError("Data contains out-of-domain values.")

        local_value_counts = _reliability_data_to_value_counts(reliability_data, value_domain)
    else:
        local_value_counts = np.asarray(value_counts)
        if value_domain is None:
            value_domain = np.arange(local_value_counts.shape[1])
        else:
            value_domain = np.asarray(value_domain)
            if local_value_counts.shape[1] != len(value_domain):
                raise ValueError("value_domain size must match number of columns in value_counts.")

    if len(value_domain) <= 1:
        raise ValueError("There must be at least two distinct values in the domain.")

    if (local_value_counts.sum(axis=-1) <= 1).all():
        raise ValueError("At least one item must be coded by two or more coders.")

    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.inexact):
        raise ValueError("`dtype` must be an inexact floating type.")

    distance_metric = _distance_metric(level_of_measurement)

    # ----- GLOBAL QUANTITIES -----
    # Observed coincidence matrix for the entire dataset
    o = _coincidences(local_value_counts, dtype=dtype)
    # Marginal frequencies
    n_v = o.sum(axis=0)
    # Random (chance) matrix
    e = _random_coincidences(n_v, dtype=dtype)
    # Distance matrix
    d = _distances(value_domain, distance_metric, n_v, dtype=dtype)

    # Global expected disagreement = sum(e * d) / sum(e)
    total_e = e.sum()
    if total_e == 0:
        # Degenerate case: if there's no variability in the entire dataset
        # we define alpha = 1 for all items
        N = local_value_counts.shape[0]
        return np.ones(N, dtype=dtype)
    global_expected_disagreement = (e * d).sum() / total_e

    # ----- PER-ITEM ALPHAS -----
    # For each item, build a small coincidence matrix and measure D_O_i
    N = local_value_counts.shape[0]
    per_item_alphas = np.zeros(N, dtype=dtype)
    for i in range(N):
        # shape (1, V) for this single item
        item_counts = local_value_counts[[i], :]
        # per-item coincidence matrix (V, V)
        item_coincidences = _coincidences(item_counts, dtype=dtype)  # sum axis=0 for just 1 item

        item_sum = item_coincidences.sum()
        if item_sum < 1e-12:
            # No real data => can't compute
            # You could define alpha = 1 or alpha = np.nan
            # For simplicity, let's define alpha=1 if no disagreement is possible
            per_item_alphas[i] = 1.0
            continue

        # D_O_i = sum_{c1,c2}(coincidence_{c1,c2} * d_{c1,c2}) / sum_{c1,c2}(coincidence_{c1,c2})
        observed_disagreement_item = (item_coincidences * d).sum() / item_sum

        # alpha_i = 1 - (D_O_i / D_E)
        # D_E is the same for all items, i.e., `global_expected_disagreement`
        if abs(global_expected_disagreement) < 1e-12:
            per_item_alphas[i] = 1.0
        else:
            per_item_alphas[i] = 1 - (observed_disagreement_item / global_expected_disagreement)

    return per_item_alphas, d
