# RQ3 code: species thermal exposure

## Setup (once)

Python 3.12. From the project folder:

```bash
python -m venv .venv
source .venv/bin/activate 
pip install -r requirements_rq3.txt
```

## Run

Put `rq3_species_exposure.py` in the project folder (next to `data/`), then:

```bash
python rq3_species_exposure.py
```

Run time is about 3–15 minutes, depending on the laptop. The script prints each table as it goes.

## Data

The script looks for three inputs in `data/`. If a file is missing, it downloads it first.

| File | What it is | If missing |
|---|---|---|
| `gdp_cc_subset.pkl` | GDP hourly drifters, 25–48°N, 135–115°W, 2012–2021 (1,412,827 obs, 341 drifters) | downloaded with `clouddrift` (slow the first time) |
| `oisst/oisst_YYYY.nc` | NOAA OISST v2.1 daily SST, 1982–2021 | downloaded year by year from NOAA ERDDAP |
| `cufes_cc.csv` | CalCOFI CUFES egg samples (sardine and anchovy egg counts, ship SST) | downloaded from NOAA ERDDAP |

## What each step does and where it appears in the report

| Step in the script | Method | Output in `figures/rq3/` | Report |
|---|---|---|---|
| 3. Heatwave threshold | Hobday et al. (2016): 90th percentile per day of year (±5 days, 31-day smoothing, 1982–2011 baseline), 5+ days in a row | used by every step below | Methods |
| B0 | Drifter and OISST temperature vs ship temperature at the egg samples | `table_B0_temperature_inputs.csv` | Data checks |
| B1 | Share of spring samples (31.5–35.5°N) in a heatwave or above each species' limit, by period, 95% CI from resampling survey days. Drifter cross-check. | `figB1_exposure_by_period.png`, `table_B1_*.csv` | Figure 3.1 table, Figure 3.2 |
| B2 | Egg presence by period, latitude where eggs were found, ship SST at the eggs | `figB2_where_fish_spawned.png`, `table_B2_*.csv` | Figure 3.3 |
| B3 | Logistic regression with inputs A (place + season), B (+ temperature range), C (+ heatwave exposure), tested leave-one-year-out | `table_B3_models.csv`, `table_B3_model_gains.csv` | Section 8.3 |
| B3b | Same inputs in three model sets: rule-based, logistic regression, gradient boosting; gain C − A; simplest set not clearly worse than the best | `figB3b_model_sets.png`, `table_B3b_*.csv` | Figure 3.4 table, Figure 3.5 |
| B4 | Sensitivity: species range moved ±2 °C, exposure window 10/30/60 days | `table_B4_*.csv` | Sensitivity |

`rq3_headline_numbers.csv` lists the key numbers. The expected values are:

- B0: 663 drifter–ship pairs, mean difference −0.08 °C, r = 0.93
- B1: in a heatwave 44% [33–55] in the Blob vs 10% [3–18] in 2019–20
- B3: logistic AUC, sardine 0.584 / 0.633 / 0.669 (A/B/C); anchovy 0.773 / 0.783 / 0.789
- B3b: gain C − A for sardine +0.015* / +0.084* / +0.029* (rule / logistic / boosting)
- B3b chosen models: sardine, gradient boosting C (AUC 0.721); anchovy, logistic regression C (AUC 0.789)

## Reproducibility

- All random steps use one generator with `SEED = 42`, called in a fixed order, so a re-run gives the same numbers.
- Gradient boosting uses `random_state = SEED`.
- The script was checked against the original analysis notebook (`RQ3_species_exposure.ipynb`). Every table matched.
