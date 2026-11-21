# AI Models for Traffic Crash Prediction on Bavarian Motorways

Code for the analysis, figures and maps of the thesis *AI Models for Traffic Crash Prediction on Bavarian Highways* and the accompanying paper.
It covers matching crash records to the Bavarian motorway network (Basis-DLM), building detector-based datasets, training positive-unlabeled (PU) and supervised crash-risk models (tabular, sequence and graph learners), and running the sensitivity and calibration analyses.

**No data is included in this repository.** All input data has to be obtained from the original providers (see below). Intermediate and final results are written to `analysis/results/` and `analysis/cache/`, which are git-ignored.

## Repository layout

```
analysis/        analysis pipeline, phase1_* ... phase13_*, v2_pipeline.py
dlm/             export of Basis-DLM road axes / motorway carriageways (export_dlm.py)
viz/             map figures (network, detectors, BASt stations)
thesis/          thesis figures and data audits (writes to thesis/image/)
paper/figures/   paper figures; short/ = short-paper versions with template fonts
run_all.sh       full rerun of the main pipeline
run_resume.sh    resume from phase 6 onwards
```

## Data

The scripts expect the following folders in the repository root (plain folders or symlinks):

| Folder | Content | Used by |
|---|---|---|
| `accidents/` | Unfallatlas crash records: `Unfallorte<year>_EPSG25832_CSV.zip` and `Unfallorte<year>_standardized.csv` | phase1, phase5, phase7, audits |
| `autobahn/Traffic`, `Traffic_2024` | motorway detector traffic data (FG1), incl. `selection_FG1.csv` | phase1, phase2, phase5, maps |
| `autobahn/Weather`, `Weather_2024` | motorway weather station data (FG3), incl. `selection_FG3.csv` | phase5, maps |
| `waze_alerts/<year>/` | Waze accident alerts | phase1 |
| `dlm/` | `road_axes.parquet`, `bab_carriageways.parquet`, created from Basis-DLM Bayern (LDBV) with `python3 dlm/export_dlm.py <by_basisdlm_plus.gpkg>` | phase1, dlm_assign |
| `h1100d/`, `h1101c/` | Bavarian road-accident statistics workbooks (`h1100d_<year>00.xlsx`, `h1101c_<year><month>.xlsx`) | phase3_class_prior |
| `bast/` | BASt hourly counts (`Jawe2024_stations.csv`, `2024_A_S.zip`) | viz/thesis_map_bast.py |
| `viz/basemap_bavaria.json` | simplified motorway axes and state outline from Basis-DLM | viz/ |
| `traffic_case_output/` | detector extracts for the three example crash hours | paper/figures/make_fig_traffic_cases.py |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`dlm/export_dlm.py` runs once with a Python that has `pyogrio` (GDAL) installed; the rest of the pipeline only needs the packages in `requirements.txt`.

## Running

```bash
PYTHON=.venv/bin/python ./run_all.sh          # full pipeline, logs to run_stage{1,2,3}.log
PYTHON=.venv/bin/python ./run_resume.sh       # resume from phase 6
```

The stages and their order are listed in `run_all.sh`. Single steps can be run directly, e.g. `python analysis/phase6_minute_pu.py`; scripts with options show them with `--help`.
Stage 2 uses a GPU (`CUDA_VISIBLE_DEVICES`, default 0).

Figures, after the pipeline has finished:

```bash
python thesis/make_figures.py                 # thesis results figures -> thesis/image/
python viz/thesis_maps.py                     # maps -> thesis/image/
python paper/figures/make_figures.py          # paper figures
```
