# polaris

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20172239.svg)](https://doi.org/10.5281/zenodo.20172239)

Research artifact. Full description will be added upon publication of the
associated manuscript.

## Citation

If you use this artifact, please cite:

```
@software{polaris_2026,
  author       = {Trinh, Van-Quynh and Huynh, Trong-Thua and
                  Huynh, De-Thu and Le, Ngoc-Hieu},
  title        = {polaris: research artifact},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.20172239},
  url          = {https://doi.org/10.5281/zenodo.20172239}
}
```

## Environment

```bash
python -m venv venv
source venv/bin/activate     # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Tested with Python 3.10.12.

## Data

Uses the publicly available
[EMBER2024 dataset](https://github.com/FutureComputing4AI/EMBER2024). Download
the parquet files and place them in a single directory referenced by the
`--parquet_dir` argument below.

A relabel frame derived from updated VirusTotal labels is used by some
configurations; see comments in `state_aware_reweighting.py` for the expected
schema.

## Quick start

```bash
# Single configuration, single seed
python pipeline_v2.py \
    --config_id baseline_B --seed 42 \
    --parquet_dir /path/to/parquet \
    --relabel_frame /path/to/relabel.parquet \
    --output_dir ./results \
    --num_threads 60 \
    --no_hpo

# Full ablation across configurations and seeds
bash scripts/run_phase4a_ablation.sh
bash scripts/run_phase4b_optuna100.sh

# Aggregate + figures
python aggregate_phase4.py --results_dir ./results --out_dir ./results/aggregated
python generate_figures.py --results_dir ./results --out_dir ./results/aggregated/figures
```

See individual module docstrings for usage details.

## Tests

```bash
python -m pytest tests/ preflight/tests/ -q
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Contact

`quynhtv@ptit.edu.vn`
