"""Loss functions and season sequence constraints for frozen-feature runs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _check_logits(logits: torch.Tensor) -> int:
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.ndim < 1:
        raise ValueError("logits must have a class dimension")
    if logits.shape[-1] < 1:
        raise ValueError("logits must contain at least one class")
    if not logits.is_floating_point():
        raise TypeError("logits must have a floating-point dtype")
    return int(logits.shape[-1])


def hard_ce(logits: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy for integer class-index targets.

    ``logits`` may have any number of leading dimensions; ``hard`` must have
    exactly those dimensions and contain class indices in the final-logit
    dimension's range.
    """
    num_classes = _check_logits(logits)
    target = torch.as_tensor(hard, device=logits.device)
    expected_shape = tuple(logits.shape[:-1])
    if tuple(target.shape) != expected_shape:
        raise ValueError(
            f"hard target shape {tuple(target.shape)} does not match logits "
            f"leading shape {expected_shape}"
        )
    if target.is_floating_point() and not torch.equal(target, target.round()):
        raise ValueError("hard targets must be integer class indices")
    if target.dtype == torch.bool:
        raise TypeError("hard targets must be integer class indices, not bool")
    flattened_logits = logits.reshape(-1, num_classes)
    flattened_target = target.to(dtype=torch.long).reshape(-1)
    return F.cross_entropy(flattened_logits, flattened_target, reduction="mean")


def soft_ce(logits: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy for probability-vector targets."""
    _check_logits(logits)
    target = torch.as_tensor(soft, device=logits.device)
    if tuple(target.shape) != tuple(logits.shape):
        raise ValueError(
            f"soft target shape {tuple(target.shape)} does not match logits "
            f"shape {tuple(logits.shape)}"
        )
    target = target.to(dtype=logits.dtype)
    per_item = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    return per_item.mean()


def ordinal_loss(logits: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Mean squared discrepancy between predicted and target class CDFs.

    The final class has no cumulative threshold of its own, so the loss
    averages over the ``K - 1`` cumulative probabilities ``P(y <= k)``.
    """
    num_classes = _check_logits(logits)
    if num_classes < 2:
        raise ValueError("ordinal_loss requires at least two ordered classes")
    target = torch.as_tensor(soft, device=logits.device)
    if tuple(target.shape) != tuple(logits.shape):
        raise ValueError(
            f"soft target shape {tuple(target.shape)} does not match logits "
            f"shape {tuple(logits.shape)}"
        )
    target = target.to(dtype=logits.dtype)
    predicted_cdf = F.softmax(logits, dim=-1).cumsum(dim=-1)[..., :-1]
    target_cdf = target.cumsum(dim=-1)[..., :-1]
    return (predicted_cdf - target_cdf).square().mean()


def _as_allowed_mask(allowed: Any, expected_shape: Tuple[int, int]) -> torch.Tensor:
    mask = torch.as_tensor(allowed)
    if mask.dtype != torch.bool:
        raise TypeError("allowed must be a boolean [T, K] mask")
    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            f"allowed mask shape {tuple(mask.shape)} does not match expected "
            f"shape {expected_shape}"
        )
    if mask.device.type != "cpu":
        mask = mask.detach().cpu()
    return mask


def _reachable_states(mask: torch.Tensor) -> List[List[int]]:
    """Return states with a valid nondecreasing prefix at each time step."""
    reachable: List[List[int]] = []
    previous: List[int] = []
    for time_index, row in enumerate(mask.tolist()):
        current = [
            state
            for state, is_allowed in enumerate(row)
            if is_allowed
            and (time_index == 0 or any(previous_state <= state for previous_state in previous))
        ]
        if not current:
            raise ValueError(
                "allowed mask has no nondecreasing state path "
                f"through time index {time_index}"
            )
        reachable.append(current)
        previous = current
    return reachable


def monotone_path_loss(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Negative mean log marginal likelihood of allowed monotone paths.

    A path may stay in its current state or jump forward by any number of
    states. It never needs to visit every intermediate state. Unreachable
    cells are omitted from the dynamic program, so no all-``-inf`` reduction
    is differentiated.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.ndim != 2:
        raise ValueError("monotone_path_loss expects logits with shape [T, K]")
    time_steps, num_classes = map(int, logits.shape)
    if time_steps < 1 or num_classes < 1:
        raise ValueError("logits must have positive time and class dimensions")
    if not logits.is_floating_point():
        raise TypeError("logits must have a floating-point dtype")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits must be finite")
    mask = _as_allowed_mask(allowed, (time_steps, num_classes))
    reachable = _reachable_states(mask)

    # Keep double precision for gradcheck and use float32 for half/bfloat16
    # inputs so the path partition calculation remains numerically stable.
    work_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    log_probabilities = F.log_softmax(logits.to(dtype=work_dtype), dim=-1)

    previous: Dict[int, torch.Tensor] = {}
    for time_index, states in enumerate(reachable):
        current: Dict[int, torch.Tensor] = {}
        for state in states:
            if time_index == 0:
                prefix_log_mass = log_probabilities.new_zeros(())
            else:
                predecessor_terms = [
                    previous[predecessor]
                    for predecessor in previous
                    if predecessor <= state
                ]
                # Structural reachability guarantees a nonempty list.
                prefix_log_mass = torch.logsumexp(torch.stack(predecessor_terms), dim=0)
            current[state] = prefix_log_mass + log_probabilities[time_index, state]
        previous = current

    log_partition = torch.logsumexp(torch.stack(list(previous.values())), dim=0)
    return -log_partition / time_steps


def _positional_indices(indices: Iterable[int], name: str, row_count: int) -> List[int]:
    array = np.asarray(list(indices))
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence of row indices")
    if array.size == 0:
        return []
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integer positional row indices")
    result = [int(value) for value in array.tolist()]
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate row indices")
    if any(value < 0 or value >= row_count for value in result):
        raise IndexError(f"{name} contains a row index outside [0, {row_count})")
    return result


def _row_value(rows: Any, position: int, column: str) -> Any:
    if hasattr(rows, "iloc") and hasattr(rows, "columns"):
        return rows[column].iloc[position]
    row = rows[position]
    if isinstance(row, dict):
        return row[column]
    if hasattr(row, "_asdict"):
        return row._asdict()[column]
    try:
        return row[column]
    except (IndexError, KeyError, TypeError):
        return getattr(row, column)


def _has_field(rows: Any, column: str) -> bool:
    if hasattr(rows, "columns"):
        return column in rows.columns
    if not len(rows):
        return False
    row = rows[0]
    if isinstance(row, dict):
        return column in row
    if hasattr(row, "_fields"):
        return column in row._fields
    return hasattr(row, column)


def _season_key_value(value: Any, name: str, position: int) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"training row {position} has a missing {name}")
    return str(value)


def _normalized_timestamp(value: Any, position: int) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"training row {position} has an invalid date") from exc
    if pd.isna(timestamp):
        raise ValueError(f"training row {position} has a missing date")
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def build_sequences(
    rows: Any,
    train_indices: Sequence[int],
    selected_indices: Sequence[int],
) -> List[Dict[str, Any]]:
    """Build chronological train-season masks from selected soft labels.

    Indices are positional offsets into ``rows``. Every training row in a
    season with at least one selected anchor is returned in date order. A
    selected row permits exactly the states with positive ``active_target``
    mass; unselected rows permit every state and their target/class fields are
    never read. Seasons without anchors are omitted.
    """
    if not hasattr(rows, "__len__"):
        raise TypeError("rows must be a sized sequence or pandas DataFrame")
    row_count = len(rows)
    train = _positional_indices(train_indices, "train_indices", row_count)
    selected = _positional_indices(selected_indices, "selected_indices", row_count)
    train_set = set(train)
    if not set(selected).issubset(train_set):
        raise ValueError("selected_indices must be a subset of train_indices")
    if not train:
        return []

    required = ("station_code", "station_year", "date")
    if hasattr(rows, "columns"):
        missing = [column for column in required if column not in rows.columns]
        if missing:
            raise ValueError(f"rows is missing required columns: {', '.join(missing)}")
    selected_set = set(selected)
    groups: Dict[Tuple[str, str], List[Tuple[pd.Timestamp, int]]] = {}
    planting_dates: Dict[Tuple[str, str], set[pd.Timestamp]] = {}
    for position in train:
        station_code = _season_key_value(
            _row_value(rows, position, "station_code"), "station_code", position
        )
        station_year = _season_key_value(
            _row_value(rows, position, "station_year"), "station_year", position
        )
        date = _normalized_timestamp(_row_value(rows, position, "date"), position)
        season_key = (station_code, station_year)
        groups.setdefault(season_key, []).append((date, position))
        if _has_field(rows, "planting_date"):
            planting_date = _row_value(rows, position, "planting_date")
            if planting_date is not None and not pd.isna(planting_date):
                planting_dates.setdefault(season_key, set()).add(
                    _normalized_timestamp(planting_date, position)
                )

    for season_key, season_planting_dates in planting_dates.items():
        if len(season_planting_dates) > 1:
            raise ValueError(
                f"station season {season_key[0]!r}/{season_key[1]!r} has "
                "multiple planting_date values"
            )

    # Duplicated dates make ordering ambiguous even for an unanchored season
    # that will later be omitted.
    for (station_code, station_year), dated_positions in groups.items():
        dates = [date for date, _ in dated_positions]
        if len(dates) != len(set(dates)):
            raise ValueError(
                f"duplicate dates in station season {station_code!r}/{station_year!r}"
            )

    # The class count is learned from selected labels only. With no anchors,
    # there is no sequence to return and no target needs to be inspected.
    class_count = None
    selected_supports: Dict[int, np.ndarray] = {}
    for position in selected:
        raw_target = _row_value(rows, position, "active_target")
        try:
            target = np.asarray(raw_target, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"selected row {position} has an invalid active_target") from exc
        if target.ndim != 1 or target.size < 1:
            raise ValueError(f"selected row {position} active_target must be a nonempty vector")
        if not np.isfinite(target).all() or (target < 0).any():
            raise ValueError(f"selected row {position} active_target must be finite and nonnegative")
        if not (target > 0).any():
            raise ValueError(f"selected row {position} active_target has empty support")
        if class_count is None:
            class_count = int(target.size)
        elif target.size != class_count:
            raise ValueError("selected active_target vectors have inconsistent class counts")
        selected_supports[position] = target > 0

    if class_count is None:
        return []

    sequences: List[Dict[str, Any]] = []
    for (station_code, station_year), dated_positions in groups.items():
        dated_positions.sort(key=lambda item: (item[0], item[1]))
        ordered_indices = [position for _, position in dated_positions]
        if not any(position in selected_set for position in ordered_indices):
            continue

        allowed = np.ones((len(ordered_indices), class_count), dtype=np.bool_)
        for time_index, position in enumerate(ordered_indices):
            if position in selected_supports:
                support = selected_supports[position]
                if support.size != class_count:
                    raise ValueError("selected active_target vectors have inconsistent class counts")
                allowed[time_index] = support

        mask = torch.as_tensor(allowed, dtype=torch.bool)
        try:
            _reachable_states(mask)
        except ValueError as exc:
            raise ValueError(
                f"inconsistent selected anchors in station season "
                f"{station_code!r}/{station_year!r}: {exc}"
            ) from exc

        sequences.append(
            {
                "indices": np.asarray(ordered_indices, dtype=np.int64),
                "allowed": mask,
                "station_code": station_code,
                "station_year": station_year,
            }
        )

    return sequences
