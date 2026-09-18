"""PSI math tests (AC-10; FR-19)."""

from __future__ import annotations

from mlops_pipeline.drift.psi import (
    DRIFT_NONE,
    bucket_proportions,
    classify_psi,
    psi_from_proportions,
)


def test_identical_distributions_psi_near_zero() -> None:
    expected = [0.1] * 10
    assert psi_from_proportions(expected, expected) == 0.0
    assert classify_psi(psi_from_proportions(expected, expected)) == DRIFT_NONE


def test_shifted_distribution_has_larger_psi() -> None:
    expected = [0.1] * 10
    shifted = [0.5, 0.3, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    psi = psi_from_proportions(expected, shifted)
    assert psi > 0.25
    assert classify_psi(psi) == "significant"


def test_moderate_shift_classified_moderate() -> None:
    expected = [0.25, 0.25, 0.25, 0.25]
    actual = [0.40, 0.25, 0.20, 0.15]  # PSI ≈ 0.133 → moderate band
    psi = psi_from_proportions(expected, actual)
    assert 0.10 <= psi < 0.25
    assert classify_psi(psi) == "moderate"


def test_bucket_proportions_sums_to_one() -> None:
    edges = [0.0, 1.0, 2.0, 3.0]
    props = bucket_proportions([0.5, 1.5, 1.6, 2.5, 5.0], edges)
    assert abs(sum(props) - 1.0) < 1e-9
    # value below first edge → first bin; value above last edge → last bin.
    assert bucket_proportions([-10.0], edges) == [1.0, 0.0, 0.0]
    assert bucket_proportions([100.0], edges) == [0.0, 0.0, 1.0]


def test_empty_bins_do_not_blow_up() -> None:
    expected = [0.5, 0.5]
    actual = [1.0, 0.0]  # empty second bin
    psi = psi_from_proportions(expected, actual)
    assert psi > 0.0  # finite, no log(0)
