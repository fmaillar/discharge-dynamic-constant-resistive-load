#!/usr/bin/env python3
"""Version 3: characterize voltage-dependent departures from global scaling.

The analysis focuses on the functional shape of the local dynamic ratio

    K(V) = (dV/dt)_{3.33 ohm} / (dV/dt)_{5 ohm}

rather than imposing discrete regimes.

Input Parquet files are immutable source data in ../parquet_discharge.
Every Parquet input is opened explicitly with mode "rb". Outputs are written
only inside this repository.

Outputs
-------
figures/scaling_v3/08_ratio_with_bootstrap_ci.pdf
figures/scaling_v3/09_ratio_derivative.pdf
figures/scaling_v3/10_significant_departure_from_alpha.pdf
figures/scaling_v3/11_parameter_sensitivity.pdf
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT.parent / "parquet_discharge"
OUTPUT_DIR = REPO_ROOT / "figures" / "scaling_v3"
DISCHARGES = tuple(range(70, 80))


@dataclass(frozen=True)
class Run:
    discharge: int
    resistance_ohm: float
    fs_hz: float
    time_s: np.ndarray
    voltage_v: np.ndarray


def read_table_read_only(path: Path, columns: list[str] | None = None):
    """Read a Parquet table through an explicitly read-only file descriptor."""
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
    groups = group_by_resistance(runs)
    matrices: dict[float, np.ndarray] = {}

    for resistance, group in groups.items():
        rows = []
        for run in group:
            v, dvdt = smooth_and_rate(run, window_s, polyorder)
            rows.append(interp_rate(v, dvdt, voltage_grid))
        matrices[resistance] = np.vstack(rows)

    return matrices


def fit_alpha(fast: np.ndarray, slow: np.ndarray, mask: np.ndarray) -> float:
    x = slow[mask]
    y = fast[mask]
    return float(np.dot(x, y) / np.dot(x, x))


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


def bootstrap_ratio(
    fast_matrix: np.ndarray,
    slow_matrix: np.ndarray,
    threshold: float,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_fast = fast_matrix.shape[0]
    n_slow = slow_matrix.shape[0]
    n_grid = fast_matrix.shape[1]

    samples = np.full((n_boot, n_grid), np.nan, dtype=float)

    for b in range(n_boot):
        fast_idx = rng.integers(0, n_fast, size=n_fast)
        slow_idx = rng.integers(0, n_slow, size=n_slow)

        fast_mean = np.nanmean(fast_matrix[fast_idx], axis=0)
        slow_mean = np.nanmean(slow_matrix[slow_idx], axis=0)
        mask = robust_mask(fast_mean, slow_mean, threshold)

        ratio = np.full(n_grid, np.nan)
        ratio[mask] = fast_mean[mask] / slow_mean[mask]
        samples[b] = ratio

    median = np.nanmedian(samples, axis=0)
    lower = np.nanpercentile(samples, 2.5, axis=0)
    upper = np.nanpercentile(samples, 97.5, axis=0)
    return median, lower, upper


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


def spline_ratio_and_derivative(
    voltage: np.ndarray,
    ratio: np.ndarray,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    spline = UnivariateSpline(voltage, ratio, s=smoothing)
    return spline(voltage), spline.derivative()(voltage)


def sensitivity_curves(
    runs: list[Run],
    voltage_grid: np.ndarray,
    windows_s: list[float],
    thresholds: list[float],
    polyorder: int,
) -> list[tuple[str, np.ndarray]]:
    curves: list[tuple[str, np.ndarray]] = []

    for window_s in windows_s:
        matrices = prepare_rate_matrix(runs, voltage_grid, window_s, polyorder)
        resistances = sorted(matrices)
        fast = np.nanmean(matrices[resistances[0]], axis=0)
        slow = np.nanmean(matrices[resistances[1]], axis=0)

        for threshold in thresholds:
            mask = robust_mask(fast, slow, threshold)
            ratio = np.full_like(voltage_grid, np.nan)
            ratio[mask] = fast[mask] / slow[mask]
            curves.append(
                (f"{window_s:g} s, {threshold:.1e} V/s", ratio)
            )

    return curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-s", type=float, default=60.0)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--grid-points", type=int, default=1000)
    parser.add_argument("--rate-threshold", type=float, default=2.0e-4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument(
        "--spline-s",
        type=float,
        default=0.01,
        help="UnivariateSpline smoothing factor on the robust ratio interval",
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

    # Define the common voltage grid from the default preprocessing.
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

    ratio = np.full_like(voltage_grid, np.nan)
    ratio[mask] = fast_mean[mask] / slow_mean[mask]

    rng = np.random.default_rng(args.seed)
    boot_median, boot_lower, boot_upper = bootstrap_ratio(
        fast_matrix,
        slow_matrix,
        args.rate_threshold,
        args.bootstrap,
        rng,
    )

    robust_slice = longest_true_slice(mask)
    v_robust = voltage_grid[robust_slice]
    ratio_robust = ratio[robust_slice]

    if np.any(~np.isfinite(ratio_robust)):
        raise ValueError("Non-finite values inside longest robust interval")

    ratio_smooth, ratio_derivative = spline_ratio_and_derivative(
        v_robust,
        ratio_robust,
        args.spline_s,
    )

    # 08: ratio and bootstrap confidence interval.
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(voltage_grid, ratio, label="Mean dynamic ratio")
    ax.plot(v_robust, ratio_smooth, linewidth=2.0, label="Spline")
    ax.fill_between(
        voltage_grid,
        boot_lower,
        boot_upper,
        alpha=0.2,
        label="95% bootstrap CI",
    )
    ax.axhline(alpha, linestyle="--", linewidth=1.0, label=f"Global α = {alpha:.3f}")
    ax.set_xlabel("Smoothed voltage [V]")
    ax.set_ylabel(
        f"(dV/dt) at {r_fast:g} Ω / (dV/dt) at {r_slow:g} Ω"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_ratio_with_bootstrap_ci.pdf")
    plt.close(fig)

    # 09: derivative of the fitted local scaling law.
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(v_robust, ratio_derivative)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Smoothed voltage [V]")
    ax.set_ylabel("dK/dV [V⁻¹]")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "09_ratio_derivative.pdf")
    plt.close(fig)

    # 10: regions where the bootstrap CI excludes the global alpha.
    significant_low = np.isfinite(boot_upper) & (boot_upper < alpha)
    significant_high = np.isfinite(boot_lower) & (boot_lower > alpha)
    significant = significant_low | significant_high

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(voltage_grid, ratio, linewidth=1.0, label="Dynamic ratio")
    ax.axhline(alpha, linestyle="--", linewidth=1.0, label=f"Global α = {alpha:.3f}")
    ax.fill_between(
        voltage_grid,
        boot_lower,
        boot_upper,
        alpha=0.15,
        label="95% bootstrap CI",
    )
    ax.scatter(
        voltage_grid[significant],
        ratio[significant],
        s=8,
        label="CI excludes global α",
    )
    ax.set_xlabel("Smoothed voltage [V]")
    ax.set_ylabel("Dynamic ratio")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "10_significant_departure_from_alpha.pdf")
    plt.close(fig)

    # 11: sensitivity to smoothing window and robust-rate threshold.
    sensitivity = sensitivity_curves(
        runs,
        voltage_grid,
        windows_s=[30.0, 60.0, 120.0],
        thresholds=[1.0e-4, 2.0e-4, 3.0e-4],
        polyorder=args.polyorder,
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for label, curve in sensitivity:
        ax.plot(voltage_grid, curve, linewidth=0.9, alpha=0.75, label=label)
    ax.axhline(alpha, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Smoothed voltage [V]")
    ax.set_ylabel("Dynamic ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "11_parameter_sensitivity.pdf")
    plt.close(fig)

    robust_ratio = ratio[mask]
    monotonic_fraction = float(np.mean(np.diff(ratio_smooth) < 0.0))
    significant_fraction = float(np.mean(significant[mask]))

    # Identify approximate voltage intervals where CI excludes alpha.
    indices = np.flatnonzero(significant & mask)
    intervals: list[tuple[float, float]] = []
    if indices.size:
        start = prev = indices[0]
        for idx in indices[1:]:
            if idx != prev + 1:
                intervals.append((voltage_grid[start], voltage_grid[prev]))
                start = idx
            prev = idx
        intervals.append((voltage_grid[start], voltage_grid[prev]))

    print(f"loaded {len(runs)} discharges from {DATA_DIR}")
    print(f"common voltage interval: {voltage_min:.6f} .. {voltage_max:.6f} V")
    print(
        f"longest robust interval: {v_robust[0]:.6f} .. "
        f"{v_robust[-1]:.6f} V"
    )
    print(f"global fitted alpha: {alpha:.6f}")
    print(f"bootstrap samples: {args.bootstrap}")
    print(f"spline smoothing factor: {args.spline_s:.6g}")
    print()
    print(f"robust ratio min: {np.nanmin(robust_ratio):.6f}")
    print(f"robust ratio max: {np.nanmax(robust_ratio):.6f}")
    print(f"fraction of spline slope < 0: {100*monotonic_fraction:.2f} %")
    print(
        "fraction of robust grid where 95% bootstrap CI excludes alpha: "
        f"{100*significant_fraction:.2f} %"
    )
    print("significant departure intervals:")
    for lo, hi in intervals:
        print(f"  {lo:.6f} .. {hi:.6f} V")

    min_deriv_idx = int(np.argmin(ratio_derivative))
    max_abs_idx = int(np.argmax(np.abs(ratio_derivative)))
    print()
    print(
        f"most negative dK/dV: {ratio_derivative[min_deriv_idx]:.6f} V^-1 "
        f"at {v_robust[min_deriv_idx]:.6f} V"
    )
    print(
        f"largest |dK/dV|: {ratio_derivative[max_abs_idx]:.6f} V^-1 "
        f"at {v_robust[max_abs_idx]:.6f} V"
    )
    print(f"figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
