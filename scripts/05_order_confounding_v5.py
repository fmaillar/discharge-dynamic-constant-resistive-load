#!/usr/bin/env python3
"""Version 5: test sequential-run confounding and leave-one-out robustness.

The two resistance conditions were acquired sequentially (D70--D74 at
3.33 ohm, D75--D79 at 5 ohm). This script quantifies within-group drift and
compares its magnitude with the voltage-dependent departure from the global
scaling law. It also evaluates the stability of the global scale factor alpha
under all 25 leave-one-out pairs.

Input Parquet files are immutable source data in ../parquet_discharge.
Every Parquet input is opened explicitly with mode "rb". Outputs are written
only inside this repository.

Outputs
-------
figures/scaling_v5/16_capacity_order_trend.pdf
figures/scaling_v5/17_within_group_dynamic_drift.pdf
figures/scaling_v5/18_drift_vs_scaling_residual.pdf
figures/scaling_v5/19_leave_one_out_alpha.pdf
figures/scaling_v5/20_order_confounding_summary.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import savgol_filter
from scipy.stats import linregress


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT.parent / "parquet_discharge"
OUTPUT_DIR = REPO_ROOT / "figures" / "scaling_v5"
DISCHARGES = tuple(range(70, 80))


@dataclass(frozen=True)
class Run:
    discharge: int
    resistance_ohm: float
    fs_hz: float
    measured_capacity_ah: float
    time_s: np.ndarray
    voltage_v: np.ndarray


def read_table_read_only(path: Path, columns: list[str] | None = None):
    """Read Parquet through an explicitly read-only binary descriptor."""
    with path.open("rb") as stream:
        return pq.read_table(stream, columns=columns)


def load_runs() -> list[Run]:
    metadata = read_table_read_only(
        DATA_DIR / "metadata.parquet",
        columns=[
            "discharge",
            "resistance_ohm",
            "fs_hz",
            "measured_capacity_ah",
        ],
    ).to_pydict()

    meta = {
        int(d): (float(r), float(fs), float(cap))
        for d, r, fs, cap in zip(
            metadata["discharge"],
            metadata["resistance_ohm"],
            metadata["fs_hz"],
            metadata["measured_capacity_ah"],
            strict=True,
        )
    }

    runs: list[Run] = []
    for discharge in DISCHARGES:
        resistance_ohm, fs_hz, capacity = meta[discharge]
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
                measured_capacity_ah=capacity,
                time_s=np.asarray(data["time_s"], dtype=float),
                voltage_v=np.asarray(data["voltage_v"], dtype=float),
            )
        )
    return runs


def group_by_resistance(runs: list[Run]) -> dict[float, list[Run]]:
    groups: dict[float, list[Run]] = {}
    for run in runs:
        groups.setdefault(run.resistance_ohm, []).append(run)
    for group in groups.values():
        group.sort(key=lambda run: run.discharge)
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

    voltage = savgol_filter(
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
    return voltage[margin:-margin], dvdt[margin:-margin]


def interp_rate(
    voltage: np.ndarray,
    dvdt: np.ndarray,
    voltage_grid: np.ndarray,
) -> np.ndarray:
    order = np.argsort(voltage)
    v = voltage[order]
    rate = dvdt[order]
    v_unique, idx = np.unique(v, return_index=True)
    rate_unique = rate[idx]
    return np.interp(voltage_grid, v_unique, rate_unique, left=np.nan, right=np.nan)


def build_rate_matrices(
    runs: list[Run],
    voltage_grid: np.ndarray,
    window_s: float,
    polyorder: int,
) -> dict[float, np.ndarray]:
    matrices: dict[float, np.ndarray] = {}
    for resistance, group in group_by_resistance(runs).items():
        rows = []
        for run in group:
            voltage, rate = smooth_and_rate(run, window_s, polyorder)
            rows.append(interp_rate(voltage, rate, voltage_grid))
        matrices[resistance] = np.vstack(rows)
    return matrices


def robust_mask(
    fast: np.ndarray,
    slow: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return (
        np.isfinite(fast)
        & np.isfinite(slow)
        & (np.abs(fast) >= threshold)
        & (np.abs(slow) >= threshold)
    )


def fit_alpha(fast: np.ndarray, slow: np.ndarray, mask: np.ndarray) -> float:
    x = slow[mask]
    y = fast[mask]
    denominator = float(np.dot(x, x))
    if denominator <= np.finfo(float).eps:
        raise ValueError("Degenerate alpha fit")
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


def linear_drift_profile(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slope per run index and p value at each voltage."""
    x = np.arange(matrix.shape[0], dtype=float)
    slopes = np.full(matrix.shape[1], np.nan)
    pvalues = np.full(matrix.shape[1], np.nan)

    for j in range(matrix.shape[1]):
        y = matrix[:, j]
        valid = np.isfinite(y)
        if np.count_nonzero(valid) >= 3:
            result = linregress(x[valid], y[valid])
            slopes[j] = result.slope
            pvalues[j] = result.pvalue

    return slopes, pvalues


def leave_one_out_alphas(
    fast_matrix: np.ndarray,
    slow_matrix: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """All 5 x 5 combinations leaving one run out of each load group."""
    values: list[float] = []

    for i in range(fast_matrix.shape[0]):
        fast = np.nanmean(np.delete(fast_matrix, i, axis=0), axis=0)
        for j in range(slow_matrix.shape[0]):
            slow = np.nanmean(np.delete(slow_matrix, j, axis=0), axis=0)
            mask = robust_mask(fast, slow, threshold)
            values.append(fit_alpha(fast, slow, mask))

    return np.asarray(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-s", type=float, default=60.0)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--grid-points", type=int, default=1000)
    parser.add_argument("--rate-threshold", type=float, default=2.0e-4)
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
    groups = group_by_resistance(runs)

    processed = {
        run.discharge: smooth_and_rate(run, args.window_s, args.polyorder)
        for run in runs
    }
    voltage_min = max(np.min(processed[r.discharge][0]) for r in runs)
    voltage_max = min(np.max(processed[r.discharge][0]) for r in runs)
    voltage_grid = np.linspace(voltage_min, voltage_max, args.grid_points)

    matrices = build_rate_matrices(
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
    robust_slice = longest_true_slice(mask)
    voltage = voltage_grid[robust_slice]

    alpha = fit_alpha(fast_mean, slow_mean, mask)
    scaled_residual = fast_mean - alpha * slow_mean

    fast_slope, fast_p = linear_drift_profile(fast_matrix)
    slow_slope, slow_p = linear_drift_profile(slow_matrix)

    # Estimated end-to-end within-group change across five sequential runs.
    fast_drift = 4.0 * fast_slope
    slow_drift = 4.0 * slow_slope

    fast_relative_drift = np.full_like(fast_drift, np.nan)
    slow_relative_drift = np.full_like(slow_drift, np.nan)
    valid_fast = np.isfinite(fast_drift) & (np.abs(fast_mean) > np.finfo(float).eps)
    valid_slow = np.isfinite(slow_drift) & (np.abs(slow_mean) > np.finfo(float).eps)
    fast_relative_drift[valid_fast] = 100.0 * fast_drift[valid_fast] / np.abs(
        fast_mean[valid_fast]
    )
    slow_relative_drift[valid_slow] = 100.0 * slow_drift[valid_slow] / np.abs(
        slow_mean[valid_slow]
    )

    # Capacity-vs-order trends.
    capacity_stats: dict[float, tuple[float, float, float]] = {}
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for resistance, group in groups.items():
        order = np.arange(len(group), dtype=float)
        capacities = np.asarray([run.measured_capacity_ah for run in group])
        result = linregress(order, capacities)
        capacity_stats[resistance] = (result.slope, result.pvalue, result.rvalue)
        ax.scatter(order + 1, capacities, label=f"{resistance:g} Ω")
        ax.plot(
            order + 1,
            result.intercept + result.slope * order,
            linewidth=1.0,
        )
    ax.set_xlabel("Run order within load group")
    ax.set_ylabel("Measured capacity [Ah]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "16_capacity_order_trend.pdf")
    plt.close(fig)

    # Relative dynamic drift profiles.
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(
        voltage,
        fast_relative_drift[robust_slice],
        label=f"{r_fast:g} Ω: D70→D74 equivalent drift",
    )
    ax.plot(
        voltage,
        slow_relative_drift[robust_slice],
        label=f"{r_slow:g} Ω: D75→D79 equivalent drift",
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Smoothed voltage [V]")
    ax.set_ylabel("Estimated within-group change [% of mean rate]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "17_within_group_dynamic_drift.pdf")
    plt.close(fig)

    # Compare possible sequential drift magnitude with the scaled inter-group residual.
    combined_drift_bound = np.sqrt(
        fast_drift**2 + (alpha * slow_drift) ** 2
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(
        voltage,
        np.abs(scaled_residual[robust_slice]),
        label="|Scaled inter-group residual|",
    )
    ax.plot(
        voltage,
        combined_drift_bound[robust_slice],
        label="Combined within-group drift magnitude",
    )
    ax.set_xlabel("Smoothed voltage [V]")
    ax.set_ylabel("Rate magnitude [V s⁻¹]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "18_drift_vs_scaling_residual.pdf")
    plt.close(fig)

    # Leave-one-out stability of alpha.
    loo_alpha = leave_one_out_alphas(
        fast_matrix,
        slow_matrix,
        args.rate_threshold,
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.scatter(np.arange(1, loo_alpha.size + 1), loo_alpha)
    ax.axhline(alpha, linestyle="--", linewidth=1.0, label=f"Full α = {alpha:.6f}")
    ax.axhline(
        r_slow / r_fast,
        linestyle=":",
        linewidth=1.0,
        label=f"Resistance ratio = {r_slow / r_fast:.6f}",
    )
    ax.set_xlabel("Leave-one-out pair")
    ax.set_ylabel("Fitted α")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "19_leave_one_out_alpha.pdf")
    plt.close(fig)

    rs = robust_slice
    residual_rms = float(np.sqrt(np.mean(scaled_residual[rs] ** 2)))
    drift_rms = float(np.sqrt(np.mean(combined_drift_bound[rs] ** 2)))
    drift_to_residual = drift_rms / residual_rms

    fast_rel_abs_median = float(np.nanmedian(np.abs(fast_relative_drift[rs])))
    slow_rel_abs_median = float(np.nanmedian(np.abs(slow_relative_drift[rs])))
    fast_rel_abs_p95 = float(np.nanpercentile(np.abs(fast_relative_drift[rs]), 95))
    slow_rel_abs_p95 = float(np.nanpercentile(np.abs(slow_relative_drift[rs]), 95))

    fast_sig_fraction = float(np.mean(fast_p[rs] < 0.05))
    slow_sig_fraction = float(np.mean(slow_p[rs] < 0.05))

    lines = [
        f"loaded discharges: {len(runs)}",
        f"robust voltage interval [V]: {voltage[0]:.6f} .. {voltage[-1]:.6f}",
        f"global fitted alpha: {alpha:.6f}",
        f"resistance ratio: {r_slow / r_fast:.6f}",
        "",
        "capacity order trends (slope per successive run):",
    ]

    for resistance in resistances:
        slope, pvalue, rvalue = capacity_stats[resistance]
        lines.append(
            f"  {resistance:g} ohm: slope={slope:+.8f} Ah/run, "
            f"p={pvalue:.6f}, r={rvalue:+.4f}, "
            f"estimated Dfirst->Dlast change={4*slope:+.8f} Ah"
        )

    lines.extend(
        [
            "",
            "dynamic within-group drift over robust interval:",
            (
                f"  {r_fast:g} ohm median |Dfirst->Dlast drift|: "
                f"{fast_rel_abs_median:.3f} % of mean rate"
            ),
            (
                f"  {r_fast:g} ohm 95th percentile |drift|: "
                f"{fast_rel_abs_p95:.3f} %"
            ),
            (
                f"  {r_slow:g} ohm median |Dfirst->Dlast drift|: "
                f"{slow_rel_abs_median:.3f} % of mean rate"
            ),
            (
                f"  {r_slow:g} ohm 95th percentile |drift|: "
                f"{slow_rel_abs_p95:.3f} %"
            ),
            (
                f"  fraction of voltage grid with within-group slope p<0.05 "
                f"({r_fast:g} ohm): {100*fast_sig_fraction:.2f} %"
            ),
            (
                f"  fraction of voltage grid with within-group slope p<0.05 "
                f"({r_slow:g} ohm): {100*slow_sig_fraction:.2f} %"
            ),
            "",
            "drift magnitude vs inter-group scaling departure:",
            f"  scaled-residual RMS: {residual_rms:.8f} V/s",
            f"  combined within-group drift RMS: {drift_rms:.8f} V/s",
            f"  drift/residual RMS ratio: {drift_to_residual:.4f}",
            "",
            "leave-one-out alpha (25 combinations):",
            f"  mean: {np.mean(loo_alpha):.6f}",
            f"  std: {np.std(loo_alpha, ddof=1):.6f}",
            f"  min: {np.min(loo_alpha):.6f}",
            f"  max: {np.max(loo_alpha):.6f}",
            f"  full-data alpha: {alpha:.6f}",
        ]
    )

    summary = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "20_order_confounding_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print(summary, end="")
    print(f"figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
