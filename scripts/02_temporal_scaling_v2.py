#!/usr/bin/env python3
"""Second-pass analysis of temporal scaling under constant resistive load.

This script tests whether the two discharge families differ mainly by a
global time-scaling factor and then quantifies voltage-dependent departures
from that scaling.

Input Parquet files are immutable source data in ../parquet_discharge.
Every Parquet input is opened explicitly with mode "rb". Outputs are written
only inside this repository.

Outputs
-------
figures/scaling_v2/04_robust_dynamic_ratio.pdf
figures/scaling_v2/05_scaled_rate_residual.pdf
figures/scaling_v2/06_scaling_deviation_zscore.pdf
figures/scaling_v2/07_piecewise_ratio_segments.pdf
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import savgol_filter


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT.parent / "parquet_discharge"
OUTPUT_DIR = REPO_ROOT / "figures" / "scaling_v2"

DISCHARGES = tuple(range(70, 80))


@dataclass(frozen=True)
class Run:
    discharge: int
    resistance_ohm: float
    fs_hz: float
    time_s: np.ndarray
    voltage_v: np.ndarray


@dataclass(frozen=True)
class GroupRates:
    resistance_ohm: float
    individual: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def read_table_read_only(path: Path, columns: list[str] | None = None):
    """Read a Parquet table through an explicitly read-only file descriptor."""
    with path.open("rb") as stream:
        return pq.read_table(stream, columns=columns)


def load_runs() -> list[Run]:
    metadata = read_table_read_only(
        DATA_DIR / "metadata.parquet",
        columns=["discharge", "resistance_ohm", "fs_hz"],
    ).to_pydict()

    meta_by_discharge = {
        int(d): (float(r), float(fs))
        for d, r, fs in zip(
            metadata["discharge"],
            metadata["resistance_ohm"],
            metadata["fs_hz"],
            strict=True,
        )
    }

    runs: list[Run] = []
    for discharge in DISCHARGES:
        resistance_ohm, fs_hz = meta_by_discharge[discharge]
        table = read_table_read_only(
            DATA_DIR / f"discharge_{discharge}.parquet",
            columns=["time_s", "voltage_v"],
        )
        data = table.to_pydict()
        runs.append(
            Run(
                discharge=discharge,
                resistance_ohm=resistance_ohm,
                fs_hz=fs_hz,
                time_s=np.asarray(data["time_s"], dtype=float),
                voltage_v=np.asarray(data["voltage_v"], dtype=float),
            )
        )

    return runs


def group_by_resistance(runs: list[Run]) -> dict[float, list[Run]]:
    groups: dict[float, list[Run]] = {}
    for run in runs:
        groups.setdefault(run.resistance_ohm, []).append(run)
    return dict(sorted(groups.items()))


def odd_window_points(window_s: float, fs_hz: float, n: int, polyorder: int) -> int:
    points = max(int(round(window_s * fs_hz)), polyorder + 2)
    if points % 2 == 0:
        points += 1
    if points > n:
        points = n if n % 2 == 1 else n - 1
    if points <= polyorder:
        raise ValueError("Savitzky-Golay window is too short for the polynomial order")
    return points


def smooth_and_rate(
    run: Run,
    window_s: float,
    polyorder: int,
) -> tuple[np.ndarray, np.ndarray]:
    window = odd_window_points(window_s, run.fs_hz, run.time_s.size, polyorder)
    dt = 1.0 / run.fs_hz

    voltage_smooth = savgol_filter(
        run.voltage_v,
        window_length=window,
        polyorder=polyorder,
        deriv=0,
        delta=dt,
        mode="interp",
    )
    dvdt = savgol_filter(
        run.voltage_v,
        window_length=window,
        polyorder=polyorder,
        deriv=1,
        delta=dt,
        mode="interp",
    )

    margin = window // 2
    if 2 * margin >= run.time_s.size:
        raise ValueError("Smoothing margin removes the complete signal")

    return voltage_smooth[margin:-margin], dvdt[margin:-margin]


def interpolate_rate_on_voltage(
    voltage_v: np.ndarray,
    dvdt_v_per_s: np.ndarray,
    voltage_grid_v: np.ndarray,
) -> np.ndarray:
    order = np.argsort(voltage_v)
    voltage_sorted = voltage_v[order]
    rate_sorted = dvdt_v_per_s[order]

    voltage_unique, unique_index = np.unique(voltage_sorted, return_index=True)
    rate_unique = rate_sorted[unique_index]

    return np.interp(
        voltage_grid_v,
        voltage_unique,
        rate_unique,
        left=np.nan,
        right=np.nan,
    )


def build_group_rates(
    runs: list[Run],
    voltage_grid: np.ndarray,
    processed: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[float, GroupRates]:
    result: dict[float, GroupRates] = {}

    for resistance, group in group_by_resistance(runs).items():
        rows = []
        for run in group:
            voltage_smooth, dvdt = processed[run.discharge]
            rows.append(
                interpolate_rate_on_voltage(voltage_smooth, dvdt, voltage_grid)
            )

        individual = np.vstack(rows)
        result[resistance] = GroupRates(
            resistance_ohm=resistance,
            individual=individual,
            mean=np.nanmean(individual, axis=0),
            std=np.nanstd(individual, axis=0, ddof=1),
        )

    return result


def fit_scale_factor(fast: np.ndarray, slow: np.ndarray, valid: np.ndarray) -> float:
    """Least-squares scale through the origin: fast ~= alpha * slow."""
    x = slow[valid]
    y = fast[valid]
    denominator = float(np.dot(x, x))
    if denominator <= np.finfo(float).eps:
        raise ValueError("Cannot fit scale factor: degenerate slow-rate vector")
    return float(np.dot(x, y) / denominator)


def longest_true_slice(mask: np.ndarray) -> slice:
    """Return the longest contiguous True interval in a boolean mask."""
    best_start = best_stop = 0
    start: int | None = None

    for i, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start > best_stop - best_start:
                best_start, best_stop = start, i
            start = None

    if best_stop <= best_start:
        raise ValueError("No contiguous valid interval found")

    return slice(best_start, best_stop)


def segment_sse(prefix_y: np.ndarray, prefix_y2: np.ndarray, i: int, j: int) -> float:
    """SSE of y[i:j] around its own mean using prefix sums."""
    n = j - i
    if n <= 0:
        return np.inf
    total = prefix_y[j] - prefix_y[i]
    total2 = prefix_y2[j] - prefix_y2[i]
    return float(total2 - total * total / n)


def best_three_segments(
    y: np.ndarray,
    min_points: int,
) -> tuple[int, int, tuple[float, float, float]]:
    """Optimal three-piece constant segmentation by total within-segment SSE."""
    n = y.size
    if n < 3 * min_points:
        raise ValueError(
            f"Need at least {3 * min_points} points for three segments, got {n}"
        )

    prefix_y = np.r_[0.0, np.cumsum(y)]
    prefix_y2 = np.r_[0.0, np.cumsum(y * y)]

    best_cost = np.inf
    best_i = best_j = -1

    for i in range(min_points, n - 2 * min_points + 1):
        left_cost = segment_sse(prefix_y, prefix_y2, 0, i)
        for j in range(i + min_points, n - min_points + 1):
            cost = (
                left_cost
                + segment_sse(prefix_y, prefix_y2, i, j)
                + segment_sse(prefix_y, prefix_y2, j, n)
            )
            if cost < best_cost:
                best_cost = cost
                best_i, best_j = i, j

    means = (
        float(np.mean(y[:best_i])),
        float(np.mean(y[best_i:best_j])),
        float(np.mean(y[best_j:])),
    )
    return best_i, best_j, means


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--window-s",
        type=float,
        default=60.0,
        help="Savitzky-Golay smoothing window in seconds (default: 60)",
    )
    parser.add_argument(
        "--polyorder",
        type=int,
        default=3,
        help="Savitzky-Golay polynomial order (default: 3)",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=1000,
        help="Number of points in the common voltage grid (default: 1000)",
    )
    parser.add_argument(
        "--rate-threshold",
        type=float,
        default=2.0e-4,
        help=(
            "Minimum absolute mean |dV/dt| required in both groups for ratio "
            "analysis, in V/s (default: 2e-4)"
        ),
    )
    parser.add_argument(
        "--segment-min-points",
        type=int,
        default=60,
        help="Minimum grid points per automatic ratio segment (default: 60)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not DATA_DIR.is_dir():
        raise SystemExit(
            f"Dataset directory not found: {DATA_DIR}\n"
            "Expected ../parquet_discharge next to the repository."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()

    processed = {
        run.discharge: smooth_and_rate(run, args.window_s, args.polyorder)
        for run in runs
    }

    voltage_min = max(np.min(processed[run.discharge][0]) for run in runs)
    voltage_max = min(np.max(processed[run.discharge][0]) for run in runs)
    voltage_grid = np.linspace(voltage_min, voltage_max, args.grid_points)

    groups = build_group_rates(runs, voltage_grid, processed)
    if len(groups) != 2:
        raise ValueError(f"Expected exactly two resistance groups, got {len(groups)}")

    r_fast, r_slow = sorted(groups)
    fast_group = groups[r_fast]
    slow_group = groups[r_slow]

    fast = fast_group.mean
    slow = slow_group.mean

    finite = np.isfinite(fast) & np.isfinite(slow)
    robust = (
        finite
        & (np.abs(fast) >= args.rate_threshold)
        & (np.abs(slow) >= args.rate_threshold)
    )

    if np.count_nonzero(robust) < 10:
        raise ValueError("Too few points remain after robust-rate masking")

    alpha_fit = fit_scale_factor(fast, slow, robust)
    duration_fast = np.mean(
        [run.time_s[-1] for run in runs if run.resistance_ohm == r_fast]
    )
    duration_slow = np.mean(
        [run.time_s[-1] for run in runs if run.resistance_ohm == r_slow]
    )
    alpha_duration = float(duration_slow / duration_fast)
    alpha_resistance = float(r_slow / r_fast)

    ratio = np.full_like(voltage_grid, np.nan)
    ratio[robust] = fast[robust] / slow[robust]
    ratio_values = ratio[robust]

    ratio_median = float(np.median(ratio_values))
    ratio_mad = float(np.median(np.abs(ratio_values - ratio_median)))

    residual = fast - alpha_fit * slow

    n_fast = fast_group.individual.shape[0]
    n_slow = slow_group.individual.shape[0]
    residual_se = np.sqrt(
        (fast_group.std ** 2) / n_fast
        + (alpha_fit ** 2) * (slow_group.std ** 2) / n_slow
    )
    zscore = np.full_like(voltage_grid, np.nan)
    zvalid = finite & np.isfinite(residual_se) & (residual_se > np.finfo(float).eps)
    zscore[zvalid] = residual[zvalid] / residual_se[zvalid]

    # Robust ratio plot.
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(voltage_grid, ratio, label="Robust dynamic ratio")
    ax.axhline(
        alpha_fit,
        linestyle="--",
        linewidth=1.0,
        label=f"Fitted α = {alpha_fit:.3f}",
    )
    ax.axhline(
        alpha_duration,
        linestyle=":",
        linewidth=1.0,
        label=f"Duration ratio = {alpha_duration:.3f}",
    )
    ax.set_xlabel("Smoothed voltage [V]")
    ax.set_ylabel(
        f"(dV/dt) at {r_fast:g} Ω / (dV/dt) at {r_slow:g} Ω"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_robust_dynamic_ratio.pdf")
    plt.close(fig)

    # Absolute deformation after removal of the best global scale factor.
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(voltage_grid, residual)
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xlabel("Smoothed voltage [V]")
    ax.set_ylabel("Scaling residual [V s⁻¹]")
    ax.set_title(f"(dV/dt)$_{{{r_fast:g}Ω}}$ - α (dV/dt)$_{{{r_slow:g}Ω}}$")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_scaled_rate_residual.pdf")
    plt.close(fig)

    # Departure from global scaling relative to repeatability.
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(voltage_grid, zscore)
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.axhline(2.0, linewidth=0.8, linestyle=":")
    ax.axhline(-2.0, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Smoothed voltage [V]")
    ax.set_ylabel("Scaling departure / standard error")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_scaling_deviation_zscore.pdf")
    plt.close(fig)

    # Automatic three-region segmentation on the longest robust interval.
    robust_slice = longest_true_slice(robust)
    v_seg = voltage_grid[robust_slice]
    ratio_seg = ratio[robust_slice]
    if np.any(~np.isfinite(ratio_seg)):
        raise ValueError("Unexpected non-finite ratio inside robust interval")

    i, j, means = best_three_segments(ratio_seg, args.segment_min_points)
    boundary_1 = float((v_seg[i - 1] + v_seg[i]) / 2.0)
    boundary_2 = float((v_seg[j - 1] + v_seg[j]) / 2.0)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(v_seg, ratio_seg, label="Robust dynamic ratio")
    ax.axvline(boundary_1, linestyle="--", linewidth=1.0)
    ax.axvline(boundary_2, linestyle="--", linewidth=1.0)

    segment_edges = [0, i, j, ratio_seg.size]
    for start, stop, mean_value in zip(
        segment_edges[:-1],
        segment_edges[1:],
        means,
        strict=True,
    ):
        ax.hlines(
            mean_value,
            v_seg[start],
            v_seg[stop - 1],
            linewidth=2.0,
        )

    ax.set_xlabel("Smoothed voltage [V]")
    ax.set_ylabel("Dynamic ratio")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_piecewise_ratio_segments.pdf")
    plt.close(fig)

    robust_v = voltage_grid[robust]
    rmse = float(np.sqrt(np.mean(residual[robust] ** 2)))
    fast_rms = float(np.sqrt(np.mean(fast[robust] ** 2)))
    relative_rmse = rmse / fast_rms

    print(f"loaded {len(runs)} discharges from {DATA_DIR}")
    print(f"common voltage interval: {voltage_min:.6f} .. {voltage_max:.6f} V")
    print(
        f"robust ratio interval(s): {robust_v.min():.6f} .. "
        f"{robust_v.max():.6f} V"
    )
    print(
        f"robust points: {np.count_nonzero(robust)}/{voltage_grid.size} "
        f"({100*np.mean(robust):.1f} %)"
    )
    print(f"rate threshold: {args.rate_threshold:.6g} V/s")
    print()
    print(f"fitted alpha (least squares): {alpha_fit:.6f}")
    print(f"duration ratio T_slow/T_fast: {alpha_duration:.6f}")
    print(f"resistance ratio R_slow/R_fast: {alpha_resistance:.6f}")
    print(f"robust median ratio: {ratio_median:.6f}")
    print(f"robust MAD ratio: {ratio_mad:.6f}")
    print(f"scaled-rate RMSE: {rmse:.8f} V/s")
    print(f"scaled-rate relative RMSE: {100*relative_rmse:.3f} %")
    print()
    print("automatic three-region segmentation:")
    print(f"  boundary 1: {boundary_1:.6f} V")
    print(f"  boundary 2: {boundary_2:.6f} V")
    print(f"  mean ratios: {means[0]:.6f}, {means[1]:.6f}, {means[2]:.6f}")
    print(f"figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
