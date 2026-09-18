# Discharge dynamics under constant resistive load

Analysis of Li-ion battery discharge trajectories under constant resistive
loads.

## Research question

Can discharge curves obtained under different constant resistive loads be
reduced to a common voltage trajectory through temporal scaling?

## Working hypothesis

For a substantial part of the discharge,

```text
dV/dt = a(R) G(V)
```

so that changing the load resistance primarily changes the traversal rate of
a common discharge trajectory. Systematic voltage-dependent departures from
this scaling would indicate load-dependent changes in the macroscopic
discharge dynamics.

## Data layout

The experimental Parquet files are deliberately kept outside this repository:

```text
parent/
├── parquet_discharge/
└── discharge_dynamic_constant_resistive_load/
```

Analysis scripts resolve the dataset as `../parquet_discharge/`.

Parquet inputs are treated as immutable source data. Inspection and analysis
code opens them explicitly in binary read-only mode and never writes into the
dataset directory.

## Initial inspection

```bash
uv sync
uv run python scripts/00_inspect_data.py
```
