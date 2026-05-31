#!/usr/bin/env python3
"""Scoring utilities for ProteinGym downstream embedding benchmarks."""

from __future__ import annotations

import numpy as np

from benchmarks.common import vector_cosine


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < values.shape[0]:
        j = i + 1
        while j < values.shape[0] and sorted_values[j] == sorted_values[i]:
            j += 1
        # Ranks are 1-based; ties receive the average rank.
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def spearmanr(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"shape mismatch: {x_arr.shape} != {y_arr.shape}")
    if x_arr.size < 2:
        raise ValueError("Spearman correlation requires at least two values")

    rx = average_ranks(x_arr)
    ry = average_ranks(y_arr)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(rx, ry) / denom)


def pearsonr(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"shape mismatch: {x_arr.shape} != {y_arr.shape}")
    if x_arr.size < 2:
        raise ValueError("Pearson correlation requires at least two values")

    x_arr = x_arr - x_arr.mean()
    y_arr = y_arr - y_arr.mean()
    denom = np.linalg.norm(x_arr) * np.linalg.norm(y_arr)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(x_arr, y_arr) / denom)


def kendall_tau_b(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"shape mismatch: {x_arr.shape} != {y_arr.shape}")
    if x_arr.size < 2:
        raise ValueError("Kendall tau requires at least two values")

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(x_arr.size - 1):
        dx = x_arr[i] - x_arr[i + 1 :]
        dy = y_arr[i] - y_arr[i + 1 :]
        sx = np.sign(dx)
        sy = np.sign(dy)

        both_nonzero = (sx != 0) & (sy != 0)
        products = sx[both_nonzero] * sy[both_nonzero]
        concordant += int(np.count_nonzero(products > 0))
        discordant += int(np.count_nonzero(products < 0))
        ties_x += int(np.count_nonzero((sx == 0) & (sy != 0)))
        ties_y += int(np.count_nonzero((sx != 0) & (sy == 0)))

    denom = np.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0.0:
        return float("nan")
    return float((concordant - discordant) / denom)


def quantile_overlap(
    predicted: list[float] | np.ndarray,
    experimental: list[float] | np.ndarray,
    *,
    fraction: float = 0.10,
    highest: bool = True,
) -> float:
    pred_arr = np.asarray(predicted, dtype=np.float64)
    exp_arr = np.asarray(experimental, dtype=np.float64)
    if pred_arr.shape != exp_arr.shape:
        raise ValueError(f"shape mismatch: {pred_arr.shape} != {exp_arr.shape}")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    k = max(1, int(np.ceil(pred_arr.size * fraction)))
    order = np.argsort if not highest else lambda values: np.argsort(values)[::-1]
    pred_top = set(order(pred_arr)[:k].tolist())
    exp_top = set(order(exp_arr)[:k].tolist())
    return float(len(pred_top & exp_top) / k)


METRIC_FUNCTIONS = {
    "spearman": spearmanr,
    "pearson": pearsonr,
    "kendall_tau_b": kendall_tau_b,
    "top10_overlap": lambda predicted, experimental: quantile_overlap(
        predicted, experimental, fraction=0.10, highest=True
    ),
    "bottom10_overlap": lambda predicted, experimental: quantile_overlap(
        predicted, experimental, fraction=0.10, highest=False
    ),
}


def compute_metrics(
    predicted: list[float] | np.ndarray,
    experimental: list[float] | np.ndarray,
    metric_names: list[str],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in metric_names:
        if name not in METRIC_FUNCTIONS:
            raise ValueError(f"unknown downstream metric: {name}")
        metrics[name] = METRIC_FUNCTIONS[name](predicted, experimental)
    return metrics


def cosine_to_wildtype(mut_embedding: np.ndarray, wildtype_embedding: np.ndarray) -> float:
    return vector_cosine(np.ravel(mut_embedding), np.ravel(wildtype_embedding))
