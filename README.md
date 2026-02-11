# BayesMMfwd.jl

Forward model for mm-source catalogs, with a Julia baseline path and a Reactant tracing/compiled path.

The main entrypoint is:

- `gen_pysides_from_original.jl`

It loads a SIDES catalog, generates SFR/magnification/flux columns, and runs benchmark timings.

## Repository Layout

- `gen_pysides_from_original.jl`: main driver and benchmark harness.
- `gen_sfr_props.jl`: SFR-related computations (`gen_sfr_props2` is currently used by the driver).
- `gen_magnification.jl`: magnification generation.
- `gen_fluxes.jl`: SED/flux/LFIR generation.
- `load_params.jl`: parser for `SIDES_from_original.par`.
- `load_sides_csv.jl`: CSV loader for SIDES catalog inputs.
- `data/`: input catalogs (`SIDES_Bethermin2017_short2.csv`, `SIDES_Bethermin2017_short.csv`).
- `SIDES_from_original.par`: parameter file consumed by the pipeline.
- `SED_finegrid_dict.h5`, `LFIR_LIR_ratio.h5`, `Psupmu_table_Bethermin17.txt`: required data assets.

## Prerequisites

- Julia with project environment support.
- Files above available at their expected paths.
- For Reactant local development in this repo, `Reactant.jl/` is a symlink to a local checkout.

## Quick Start

From repo root:

```bash
julia --project=. gen_pysides_from_original.jl
```

This command:

1. Loads parameters and catalog input.
2. Preallocates output columns on the catalog.
3. Runs benchmark process paths.
4. Writes a formatted benchmark summary to a text file.

## Choosing Dataset Size

In `gen_pysides_from_original.jl`, update `csv_idl_path` near the bottom:

- Small/debug run: `data/SIDES_Bethermin2017_short2.csv`
- Large benchmark run: `data/SIDES_Bethermin2017_short.csv`

## Benchmark Outputs

The script writes a formatted benchmark report via `format_benchmark_report(...)`.

Typical output files used in this repo:

- `benchmark_results_small.txt`
- `benchmark_results_large.txt`

The report includes:

- dataset path
- backend (`cpu`)
- seed
- timing table
- derived ratios

## Configuration and Reproducibility

Important knobs in `gen_pysides_from_original.jl`:

- `BENCHMARK_SEED = 1234`
- `Reactant.set_default_backend("cpu")`
- parameter file path: `"SIDES_from_original.par"`

If you are comparing runs, keep seed and dataset fixed.

## Troubleshooting (Reactant)

### Scalar indexing is disallowed

Use `@allowscalar` only where scalar get/set is required (for example explicit index reads/writes inside traced loops).

### `findall` on traced boolean masks fails

Prefer mask-based formulations with `ifelse.(mask, a, b)` rather than `findall(mask)` + indexed writes.

### Trace/compile timings look too small

Compiled/traced calls can dispatch asynchronously; if timings look unrealistically tiny, ensure your timing method includes synchronization.

### Missing data file/path errors

Check:

- `SIDES_from_original.par`
- `data/SIDES_Bethermin2017_short*.csv`
- `SED_finegrid_dict.h5`
- `LFIR_LIR_ratio.h5`
- `Psupmu_table_Bethermin17.txt`

## Development Notes

- This repo currently contains several MWE scripts (`mwe_*.jl`) used for Reactant debugging.
- Keep helper scripts focused; most orchestration should stay in `gen_pysides_from_original.jl`.
