#!/usr/bin/env python3
"""Version 4: monotone scaling law and article-ready summary metrics.

The local dynamic ratio is

    K(V) = (dV/dt)_{3.33 ohm} / (dV/dt)_{5 ohm}

Version 4 replaces the unconstrained spline by a monotone non-increasing
estimate using isotonic regression (PAVA), adds a bootstrap confidence band
for that monotone law, removes all-NaN bootstrap warnings, and reports compact
summary quantities suitable for a manuscript.

Input Parquet files are immutable source data in ../parquet_discharge.
Every Parquet input is opened explicitly with mode "rb". Outputs are written
only inside this repository.

Outputs
-------
figures/scaling_v4/12_monotone_ratio_bootstrap.pdf
figures/scaling_v4/13_relative_departure_from_alpha.pdf
figures/scaling_v4/14_monotone_local_slope.pdf
figures/scaling_v4/15_summary_metrics.txt
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
OUTPUT_DIR = REPO_ROOT / "figures" / "scaling_v4"
DISCHARGES = tuple(range(70, 80))


@dataclass(frozen=True)
class Run:
    discharge: int
    resistance_ohm: float
    fs_hz: float
    time_s: np.ndarray
    voltage_v: np.ndarray


def read_table_read_only(path: Path, columns: list[str] | None = None):
    """Read Parquet through an explicitly read-only binary file descriptor."""
    with path.open("rb") as stream:
        return pq.read_table(stream, columns=columns)


def load_runs() -> list[Run]:
    metadata = read_table_read_only(
        DATA_DIR / "metadata.parquet",
        columns=["discharge", "resistance_ohm", "fs_hz"],
    ).to_pydict()

    meta = {
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
        resistance_ohm, fs_hz = meta[discharge]
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
        points = n if n % 2 else n - 1
    if points <= polyorder:
        raise ValueError("Savitzky-Golay window is too short")
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
    return voltage_smooth[margin:-margin], dvdt[margin:-margin]


def interp_rate(
    voltage_v: np.ndarray,
    dvdt: np.ndarray,
    voltage_grid: np.ndarray,
) -> np.ndarray:
    order = np.argsort(voltage_v)
    v = voltage_v[order]
    r = dvdt[order]
    v_unique, idx = np.unique(v, return_index=True)
    r_unique = r[idx]
    return np.interp(voltage_grid, v_unique, r_unique, left=np.nan, right=np.nan)


def prepare_rate_matrix(
    runs: list[Run],
    voltage_grid: np.ndarray,
    window_s: float,
    polyorder: int,
) -> dict[float, np.ndarray]:
    matrices: dict[float, np.ndarray] = {}
    for resistance, group in group_by_resistance(runs).items():
        rows = []
        for run in group:
            v, dvdt = smooth_and_rate(run, window_s, polyorder)
            rows.append(interp_rate(v, dvdt, voltage_grid))
        matrices[resistance] = np.vstack(rows)
    return matrices


def robust_mask(
    fast_mean: np.ndarray,
    slow_mean: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return (
        np.isfinite(fast_mean)
        & np.isfinite(slow_mean)
        & (np.abs(fast_mean) >= threshold)
        & (np.abs(slow_mean) >= threshold)
    )


def fit_alpha(fast: np.ndarray, slow: np.ndarray, mask: np.ndarray) -> float:
    x = slow[mask]
    y = fast[mask]
    denominator = float(np.dot(x, x))
    if denominator <= np.finfo(float).eps:
        raise ValueError("Degenerate least-squares fit")
    return float(np.dot(x, y) / denominator)


def longest_true_slice(mask: np.ndarray) -> slice:
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
        raise ValueError("No contiguous robust interval")
    return slice(best_start, best_stop)


def isotonic_increasing(y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted PAVA for a non-decreasing sequence."""
    y = np.asarray(y, dtype=float)
    if weights is None:
        weights = np.ones_like(y)
    else:
        weights = np.asarray(weights, dtype=float)

    values = y.copy()
    w = weights.copy()
    starts = np.arange(y.size)
    ends = np.arange(y.size)
    m = y.size
    i = 0

    while i < m - 1:
        if values[i] <= values[i + 1]:
            i += 1
            continue

        new_w = w[i] + w[i + 1]
        new_value = (w[i] * values[i] + w[i + 1] * values[i + 1]) / new_w

        values[i] = new_value
        w[i] = new_w
        ends[i] = ends[i + 1]

        values[i + 1 : m - 1] = values[i + 2 : m]
        w[i + 1 : m - 1] = w[i + 2 : m]
        starts[i + 1 : m - 1] = starts[i + 2 : m]
        ends[i + 1 : m - 1] = ends[i + 2 : m]
        m -= 1

        if i > 0:
            i -= 1

    fitted = np.empty_like(y)
    for k in range(m):
        fitted[starts[k] : ends[k] + 1] = values[k]
    return fitted


def isotonic_decreasing(y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted PAVA for a non-increasing sequence."""
    return -isotonic_increasing(-np.asarray(y, dtype=float), weights)


def ratio_standard_error(
    fast_matrix: np.ndarray,
    slow_matrix: np.ndarray,
) -> np.ndarray:
    """Delta-method standard error of a ratio of group means."""
    fast_mean = np.nanmean(fast_matrix, axis=0)
    slow_mean = np.nanmean(slow_matrix, axis=0)
    fast_var_mean = np.nanvar(fast_matrix, axis=0, ddof=1) / fast_matrix.shape[0]
    slow_var_mean = np.nanvar(slow_matrix, axis=0, ddof=1) / slow_matrix.shape[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        variance = (
            fast_var_mean / (slow_mean**2)
            + (fast_mean**2) * slow_var_mean / (slow_mean**4)
        )
    return np.sqrt(variance)


def bootstrap_monotone_ratio(
    fast_matrix: np.ndarray,
    slow_matrix: np.ndarray,
    robust_slice: slice,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap a monotone ratio on a fixed robust interval.

    No thresholding is repeated inside bootstrap replicates. The robust
    interval is fixed from the original sample, which avoids all-NaN columns
    and yields a directly comparable functional confidence band.
    """
    fast_part = fast_matrix[:, robust_slice]
    slow_part = slow_matrix[:, robust_slice]

    n_fast = fast_part.shape[0]
    n_slow = slow_part.shape[0]
    n_grid = fast_part.shape[1]

    samples = np.empty((n_boot, n_grid), dtype=float)

    for b in range(n_boot):
        fi = rng.integers(0, n_fast, size=n_fast)
        si = rng.integers(0, n_slow, size=n_slow)

        fast_mean = np.mean(fast_part[fi], axis=0)
        slow_mean = np.mean(slow_part[si], axis=0)

        ratio = fast_mean / slow_mean
        samples[b] = isotonic_decreasing(ratio)

    median = np.median(samples, axis=0)
    lower = np.percentile(samples, 2.5, axis=0)
    upper = np.percentile(samples, 97.5, axis=0)
    return median, lower, upper


def crossing_voltage(
    voltage: np.ndarray,
    fitted: np.ndarray,
    target: float,
) -> float | None:
    diff = fitted - target
    exact = np.flatnonzero(diff == 0.0)
    if exact.size:
        return float(voltage[exact[0]])

    crossings = np.flatnonzero(diff[:-1] * diff[1:] < 0.0)
    if crossings.size == 0:
        return None

    i = int(crossings[0])
    x0, x1 = voltage[i], voltage[i + 1]
    y0, y1 = diff[i], diff[i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def broad_range_slopes(voltage: np.ndarray, fitted: np.ndarray) -> list[tuple[float, float, float]]:
    edges = np.linspace(voltage[0], voltage[-1], 4)
    result: list[tuple[float, float, float]] = []

    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (voltage >= lo) & (voltage <= hi)
        x = voltage[mask]
        y = fitted[mask]
        if x.size < 2:
            slope = np.nan
        else:
            slope = float(np.polyfit(x, y, 1)[0])
        result.append((float(lo), float(hi), slope))

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-s", type=float, default=60.0)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--grid-points", type=int, default=1000)
    parser.add_argument("--rate-threshold", type=float, default=2.0e-4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260918)
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
    voltage_min = max(np.min(processed[r.discharge][0]) for r in runs)
    voltage_max = min(np.max(processed[r.discharge][0]) for r in runs)
    voltage_grid = np.linspace(voltage_min, voltage_max, args.grid_points)

    matrices = prepare_rate_matrix(
        runs,
        voltage_grid,
        args.window_s,
        args.polyorder,
    )

    resistances = sorted(matrices)
    if len(resistances) != 2:
        raise ValueError("Expected exactly two resistance groups")

    r_fast, r_slow = resistances
    fast_matrix = matrices[r_fast]
    slow_matrix = matrices[r_slow]

    fast_mean = np.nanmean(fast_matrix, axis=0)
    slow_mean = np.nanmean(slow_matrix, axis=0)
    mask = robust_mask(fast_mean, slow_mean, args.rate_threshold)
    alpha = fit_alpha(fast_mean, slow_mean, mask)

    robust_slice = longest_true_slice(mask)
    voltage = voltage_grid[robust_slice]
    fast = fast_mean[robust_slice]
    slow = slow_mean[robust_slice]
    raw_ratio = fast / slow

    ratio_se_full = ratio_standard_error(fast_matrix, slow_matrix)
    ratio_se = ratio_se_full[robust_slice]

    weights = np.ones_like(raw_ratio)
    finite_se = np.isfinite(ratio_se) & (ratio_se > np.finfo(float).eps)
    if np.any(finite_se):
        weights[finite_se] = 1.0 / (ratio_se[finite_se] ** 2)
        cap = np.nanpercentile(weights[finite_se], 95.0)
        weights = np.minimum(weights, cap)

    monotone = isotonic_decreasing(raw_ratio, weights)

    rng = np.random.default_rng(args.seed)
    boot_median, boot_lower, boot_upper = bootstrap_monotone_ratio(
        fast_matrix,
        slow_matrix,
        robust_slice,
        args.bootstrap,
        rng,
    )

    relative_departure_pct = 100.0 * (monotone / alpha - 1.0)

    # A smooth derivative is used only to summarize where the monotone law
    # changes fastest. It is not interpreted as a separate physical observable.
    derivative_window = min(101, monotone.size if monotone.size % 2 else monotone.size - 1)
    derivative_window = max(derivative_window, 5)
    if derivative_window % 2 == 0:
        derivative_window -= 1
    dv = float(np.mean(np.diff(voltage)))
    monotone_smoothed = savgol_filter(
        monotone,
        window_length=derivative_window,
        polyorder=2,
        mode="interp",
    )
    local_slope = savgol_filter(
        monotone,
        window_length=derivative_window,
        polyorder=2,
        deriv=1,
        delta=dv,
        mode="interp",
    )

    cross = crossing_voltage(voltage, monotone, alpha)
    slopes = broad_range_slopes(voltage, monotone)

    max_high_idx = int(np.argmax(relative_departure_pct))
    max_low_idx = int(np.argmin(relative_departure_pct))
    steep_idx = int(np.argmin(local_slope))

    # 12: monotone law with bootstrap band.
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(voltage, raw_ratio, linewidth=0.8, alpha=0.45, label="Raw mean ratio")
    ax.plot(voltage, monotone, linewidth=2.0, label="Monotone estimate")
    ax.fill_between(
        voltage,
        boot_lower,
        boot_upper,
        alpha=0.2,
        label="95% bootstrap CI",
    )
    ax.axhline(alpha, linestyle="--", linewidth=1.0, label=f"Global α = {alpha:.3f}")
    if cross is not None:
        ax.axvline(cross, linestyle=":", linewidth=1.0, label=f"Crossing = {cross:.3f} V")
    ax.set_xlabel("Smoothed voltage [V]")
    ax.set_ylabel(
        f"(dV/dt) at {r_fast:g} Ω / (dV/dt) at {r_slow:g} Ω"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "12_monotone_ratio_bootstrap.pdf")
    plt.close(fig)

    # 13: relative local departure from the global scaling law.
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(voltage, relative_departure_pct)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Smoothed voltage [V]")
    ax.set_ylabel("Local departure from global α [%]")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "13_relative_departure_from_alpha.pdf")
    plt.close(fig)

    # 14: smoothed local slope of the monotone law.
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(voltage, local_slope)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.axvline(
        voltage[steep_idx],
        linestyle=":",
        linewidth=1.0,
        label=f"Steepest decrease = {voltage[steep_idx]:.3f} V",
    )
    ax.set_xlabel("Smoothed voltage [V]")
    ax.set_ylabel("dK/dV [V⁻¹]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "14_monotone_local_slope.pdf")
    plt.close(fig)

    duration_fast = np.mean(
        [run.time_s[-1] for run in runs if run.resistance_ohm == r_fast]
    )
    duration_slow = np.mean(
        [run.time_s[-1] for run in runs if run.resistance_ohm == r_slow]
    )
    duration_ratio = float(duration_slow / duration_fast)
    resistance_ratio = float(r_slow / r_fast)

    rms_departure_pct = float(np.sqrt(np.mean(relative_departure_pct**2)))
    median_abs_departure_pct = float(np.median(np.abs(relative_departure_pct)))

    lines = [
        f"loaded discharges: {len(runs)}",
        f"robust voltage interval [V]: {voltage[0]:.6f} .. {voltage[-1]:.6f}",
        f"global fitted alpha: {alpha:.6f}",
        f"duration ratio T_slow/T_fast: {duration_ratio:.6f}",
        f"resistance ratio R_slow/R_fast: {resistance_ratio:.6f}",
        f"alpha vs resistance ratio relative error [%]: {100*(alpha/resistance_ratio-1):.4f}",
        f"bootstrap samples: {args.bootstrap}",
        "",
        f"monotone K(V) min: {np.min(monotone):.6f}",
        f"monotone K(V) max: {np.max(monotone):.6f}",
        (
            f"crossing voltage K(V)=alpha [V]: {cross:.6f}"
            if cross is not None
            else "crossing voltage K(V)=alpha [V]: none"
        ),
        f"RMS local departure from alpha [%]: {rms_departure_pct:.3f}",
        f"median absolute local departure from alpha [%]: {median_abs_departure_pct:.3f}",
        (
            f"maximum positive departure [%]: {relative_departure_pct[max_high_idx]:.3f} "
            f"at {voltage[max_high_idx]:.6f} V"
        ),
        (
            f"maximum negative departure [%]: {relative_departure_pct[max_low_idx]:.3f} "
            f"at {voltage[max_low_idx]:.6f} V"
        ),
        (
            f"steepest monotone-law decrease dK/dV [V^-1]: "
            f"{local_slope[steep_idx]:.6f} at {voltage[steep_idx]:.6f} V"
        ),
        "",
        "broad-range linear slopes of monotone K(V):",
    ]

    for lo, hi, slope in slopes:
        lines.append(f"  {lo:.6f} .. {hi:.6f} V: {slope:.6f} V^-1")

    summary = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "15_summary_metrics.txt").write_text(summary, encoding="utf-8")

    print(summary, end="")
    print(f"figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
