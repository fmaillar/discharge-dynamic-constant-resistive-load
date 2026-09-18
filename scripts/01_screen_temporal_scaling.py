#!/usr/bin/env python3
"""Screen temporal-scaling invariance of Li-ion discharges.

Input Parquet files are external immutable source data located in
../parquet_discharge relative to the repository root. Every Parquet input is
opened explicitly with mode "rb". The script writes figures only inside this
repository.

Outputs
-------
figures/screening/01_normalized_time.pdf
figures/screening/02_voltage_rate_vs_voltage.pdf
figures/screening/03_dynamic_ratio_vs_voltage.pdf
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
OUTPUT_DIR = REPO_ROOT / "figures" / "screening"

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # Avoid the half-window region where derivative estimates are least robust.
    margin = window // 2
    if 2 * margin >= run.time_s.size:
        raise ValueError("Smoothing margin removes the complete signal")

    return (
        run.time_s[margin:-margin],
        voltage_smooth[margin:-margin],
        dvdt[margin:-margin],
    )


def interpolate_rate_on_voltage(
    voltage_v: np.ndarray,
    dvdt_v_per_s: np.ndarray,
    voltage_grid_v: np.ndarray,
) -> np.ndarray:
    """Interpolate dV/dt as a function of smoothed voltage.

    The macroscopic discharge should be nearly monotonic after smoothing.
    Sorting by voltage makes the interpolation robust to the remaining tiny
    local reversals without modifying the source data.
    """
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


def group_by_resistance(runs: list[Run]) -> dict[float, list[Run]]:
    groups: dict[float, list[Run]] = {}
    for run in runs:
        groups.setdefault(run.resistance_ohm, []).append(run)
    return dict(sorted(groups.items()))


def plot_normalized_time(runs: list[Run]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for resistance, group in group_by_resistance(runs).items():
        for i, run in enumerate(group):
            normalized_time = run.time_s / run.time_s[-1]
            label = f"{resistance:g} Ω" if i == 0 else None
            ax.plot(normalized_time, run.voltage_v, linewidth=0.8, alpha=0.7, label=label)

    ax.set_xlabel("Normalized discharge time")
    ax.set_ylabel("Voltage [V]")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_normalized_time.pdf")
    plt.close(fig)


def plot_dynamic_screening(
    runs: list[Run],
    window_s: float,
    polyorder: int,
    grid_points: int,
) -> None:
    processed: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        run.discharge: smooth_and_rate(run, window_s, polyorder) for run in runs
    }

    voltage_min = max(np.min(processed[run.discharge][1]) for run in runs)
    voltage_max = min(np.max(processed[run.discharge][1]) for run in runs)
    if not voltage_min < voltage_max:
        raise ValueError("No common voltage interval across all discharges")

    voltage_grid = np.linspace(voltage_min, voltage_max, grid_points)
    groups = group_by_resistance(runs)

    group_rates: dict[float, np.ndarray] = {}

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for resistance, group in groups.items():
        rates = []
        for run in group:
            _, voltage_smooth, dvdt = processed[run.discharge]
            positive_fraction = float(np.mean(dvdt > 0.0))
            print(
                f"D{run.discharge}: R={resistance:g} ohm, "
                f"fraction(dV/dt > 0)={positive_fraction:.5f}"
            )
            rates.append(
                interpolate_rate_on_voltage(voltage_smooth, dvdt, voltage_grid)
            )

        stacked = np.vstack(rates)
        mean_rate = np.nanmean(stacked, axis=0)
        std_rate = np.nanstd(stacked, axis=0, ddof=1)
        group_rates[resistance] = mean_rate

        ax.plot(voltage_grid, mean_rate, label=f"{resistance:g} Ω")
        ax.fill_between(
            voltage_grid,
            mean_rate - std_rate,
            mean_rate + std_rate,
            alpha=0.2,
        )

    ax.set_xlabel("Smoothed voltage [V]")
    ax.set_ylabel("dV/dt [V s⁻¹]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_voltage_rate_vs_voltage.pdf")
    plt.close(fig)

    if len(group_rates) != 2:
        raise ValueError(
            f"Dynamic ratio requires exactly two resistance groups, got {len(group_rates)}"
        )

    r_fast, r_slow = sorted(group_rates)
    fast = group_rates[r_fast]
    slow = group_rates[r_slow]

    valid = (
        np.isfinite(fast)
        & np.isfinite(slow)
        & (np.abs(slow) > np.finfo(float).eps)
    )
    ratio = np.full_like(voltage_grid, np.nan)
    ratio[valid] = fast[valid] / slow[valid]

    finite_ratio = ratio[np.isfinite(ratio)]
    median_ratio = float(np.median(finite_ratio))
    mad_ratio = float(np.median(np.abs(finite_ratio - median_ratio)))
    nominal_resistance_ratio = r_slow / r_fast

    print()
    print(f"common voltage interval: {voltage_min:.6f} .. {voltage_max:.6f} V")
    print(f"median dynamic ratio: {median_ratio:.6f}")
    print(f"MAD dynamic ratio: {mad_ratio:.6f}")
    print(f"nominal resistance ratio R_slow/R_fast: {nominal_resistance_ratio:.6f}")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(voltage_grid, ratio, label="Measured dynamic ratio")
    ax.axhline(
        median_ratio,
        linestyle="--",
        linewidth=1.0,
        label=f"Median = {median_ratio:.3f}",
    )
    ax.axhline(
        nominal_resistance_ratio,
        linestyle=":",
        linewidth=1.0,
        label=f"R ratio = {nominal_resistance_ratio:.3f}",
    )
    ax.set_xlabel("Smoothed voltage [V]")
    ax.set_ylabel(
        f"(dV/dt) at {r_fast:g} Ω / (dV/dt) at {r_slow:g} Ω"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_dynamic_ratio_vs_voltage.pdf")
    plt.close(fig)


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

    print(f"loaded {len(runs)} discharges from {DATA_DIR}")
    for resistance, group in group_by_resistance(runs).items():
        durations = np.asarray([run.time_s[-1] for run in group])
        print(
            f"R={resistance:g} ohm: n={len(group)}, "
            f"duration={durations.mean():.1f} ± {durations.std(ddof=1):.1f} s"
        )

    plot_normalized_time(runs)
    plot_dynamic_screening(
        runs,
        window_s=args.window_s,
        polyorder=args.polyorder,
        grid_points=args.grid_points,
    )

    print(f"figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
