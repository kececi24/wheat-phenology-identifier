from itertools import product

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from Loss_BBCH.losses import (
    build_sequences,
    hard_ce,
    monotone_path_loss,
    ordinal_loss,
    soft_ce,
)


def _enumerated_path_loss(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    paths = []
    for states in product(range(logits.shape[1]), repeat=logits.shape[0]):
        if any(states[index] > states[index + 1] for index in range(len(states) - 1)):
            continue
        if not all(bool(allowed[time, state]) for time, state in enumerate(states)):
            continue
        paths.append(torch.stack([log_probs[time, state] for time, state in enumerate(states)]).sum())
    if not paths:
        raise ValueError("no allowed path")
    return -torch.logsumexp(torch.stack(paths), dim=0) / logits.shape[0]


def test_monotone_path_matches_brute_force_values_and_gradients():
    allowed = torch.tensor(
        [[True, True, False], [False, True, True], [False, True, True]],
        dtype=torch.bool,
    )
    logits = torch.tensor(
        [[0.2, -0.3, 0.5], [-0.4, 0.7, 0.1], [0.3, -0.2, 0.8]],
        dtype=torch.float64,
        requires_grad=True,
    )
    actual = monotone_path_loss(logits, allowed)
    expected = _enumerated_path_loss(logits, allowed)
    actual_gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, logits)[0]

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_singleton_hard_anchor_path_reduces_to_mean_ce():
    logits = torch.tensor(
        [[0.8, -0.2, 0.4], [-0.1, 0.9, 0.2], [-0.3, 0.5, 0.6]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 1, 1])
    one_hot = F.one_hot(labels, num_classes=logits.shape[-1]).to(dtype=logits.dtype)
    singleton_mask = one_hot.bool()

    ce = hard_ce(logits, labels)
    torch.testing.assert_close(soft_ce(logits, one_hot), ce)
    torch.testing.assert_close(monotone_path_loss(logits, singleton_mask), ce)


def test_free_monotone_path_allows_skipping_stages():
    logits = torch.tensor([[0.1, 0.6, -0.2, 0.8], [0.5, -0.4, 0.7, 0.0]], dtype=torch.float64)
    allowed = torch.ones_like(logits, dtype=torch.bool)

    actual = monotone_path_loss(logits, allowed)
    expected = _enumerated_path_loss(logits, allowed)

    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected)


def test_infeasible_or_empty_allowed_mask_fails():
    logits = torch.zeros((2, 3), dtype=torch.float64)
    decreasing_only = torch.tensor([[False, False, True], [False, True, False]])
    empty_row = torch.tensor([[True, True, True], [False, False, False]])

    with pytest.raises(ValueError, match="no nondecreasing state path"):
        monotone_path_loss(logits, decreasing_only)
    with pytest.raises(ValueError, match="no nondecreasing state path"):
        monotone_path_loss(logits, empty_row)


def test_backward_is_finite_when_allowed_states_are_unreachable():
    logits = torch.tensor(
        [[0.2, -0.4, 0.8], [0.5, 0.1, -0.2], [-0.3, 0.7, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    # At t=1, states 0 and 1 are allowed but unreachable after the forced 2.
    allowed = torch.tensor(
        [[False, False, True], [True, True, True], [False, True, True]],
        dtype=torch.bool,
    )

    loss = monotone_path_loss(logits, allowed)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_ordinal_loss_is_mean_squared_cdf_error_over_k_minus_one():
    probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
    target = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float64)

    value = ordinal_loss(probabilities.log(), target)

    # Predicted CDF is [.1, .3, .6], target CDF is [0, 0, 1].
    expected = torch.tensor((0.1**2 + 0.3**2 + 0.4**2) / 3, dtype=torch.float64)
    torch.testing.assert_close(value, expected)
    torch.testing.assert_close(ordinal_loss(probabilities.log(), probabilities), torch.zeros_like(value))


def _sequence_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"station_code": "A", "station_year": "A_2020", "date": "2020-01-02", "planting_date": "2019-10-01", "active_target": [1, 0, 0], "label": "stale-0"},
            {"station_code": "A", "station_year": "A_2020", "date": "2020-01-01", "planting_date": "2019-10-01", "active_target": [1, 0, 0], "label": "anchor-0"},
            {"station_code": "B", "station_year": "B_2020", "date": "2020-01-01", "planting_date": "2019-10-02", "active_target": [0, 1, 0], "label": "unanchored"},
            {"station_code": "A", "station_year": "A_2020", "date": "2020-01-03", "planting_date": "2019-10-01", "active_target": [0, 0, 1], "label": "stale-2"},
            {"station_code": "A", "station_year": "A_2021", "date": "2021-01-01", "planting_date": "2020-10-01", "active_target": [0, 0, 1], "label": "anchor-2"},
            {"station_code": "A", "station_year": "A_2021", "date": "2021-01-02", "planting_date": "2020-10-01", "active_target": [1, 0, 0], "label": "stale-year"},
        ]
    )


def test_build_sequences_orders_and_separates_anchored_seasons_only():
    rows = _sequence_rows()

    sequences = build_sequences(rows, train_indices=range(6), selected_indices=[1, 4])

    assert [(item["station_code"], item["station_year"]) for item in sequences] == [
        ("A", "A_2020"),
        ("A", "A_2021"),
    ]
    np.testing.assert_array_equal(sequences[0]["indices"], [1, 0, 3])
    np.testing.assert_array_equal(sequences[1]["indices"], [4, 5])
    assert sequences[0]["allowed"].dtype == torch.bool
    assert sequences[0]["allowed"].tolist() == [
        [True, False, False],
        [True, True, True],
        [True, True, True],
    ]
    assert sequences[1]["allowed"].tolist() == [[False, False, True], [True, True, True]]


def test_unselected_class_and_target_changes_do_not_change_allowed_masks():
    original = _sequence_rows()
    changed = original.copy(deep=True)
    for index in (0, 2, 3, 5):
        changed.at[index, "active_target"] = [0.0, 1.0, 0.0]
        changed.at[index, "label"] = "changed-unselected-class"

    before = build_sequences(original, train_indices=range(6), selected_indices=[1, 4])
    after = build_sequences(changed, train_indices=range(6), selected_indices=[1, 4])

    assert len(before) == len(after)
    for left, right in zip(before, after):
        np.testing.assert_array_equal(left["indices"], right["indices"])
        torch.testing.assert_close(left["allowed"], right["allowed"])


def test_build_sequences_rejects_selected_heldout_rows():
    rows = _sequence_rows()

    with pytest.raises(ValueError, match="subset of train_indices"):
        build_sequences(rows, train_indices=[0], selected_indices=[1])


def test_build_sequences_rejects_inconsistent_anchors_duplicate_dates_and_cycles():
    rows = _sequence_rows().iloc[:2].copy()
    rows.iloc[0, rows.columns.get_loc("date")] = "2020-01-02"
    rows.iloc[1, rows.columns.get_loc("date")] = "2020-01-01"
    rows.at[0, "active_target"] = [1, 0, 0]
    rows.at[1, "active_target"] = [0, 0, 1]
    with pytest.raises(ValueError, match="inconsistent selected anchors"):
        build_sequences(rows, train_indices=[0, 1], selected_indices=[0, 1])

    duplicate_dates = rows.copy()
    duplicate_dates.iloc[1, duplicate_dates.columns.get_loc("date")] = "2020-01-02"
    with pytest.raises(ValueError, match="duplicate dates"):
        build_sequences(duplicate_dates, train_indices=[0, 1], selected_indices=[0])

    multiple_plantings = _sequence_rows().iloc[:2].copy()
    multiple_plantings.at[0, "planting_date"] = "2019-09-01"
    with pytest.raises(ValueError, match="multiple planting_date"):
        build_sequences(multiple_plantings, train_indices=[0, 1], selected_indices=[1])
