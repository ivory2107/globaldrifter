"""
RQ3 - Did marine heatwaves overlap fish spawning habitat, and does heatwave
exposure help predict where sardine and anchovy eggs were found?

Run from the project folder:
    python rq3_species_exposure.py

Inputs (in data/):
    gdp_cc_subset.pkl   GDP hourly drifters, 25-48N, 135-115W, 2012-2021
                        (downloaded with clouddrift if missing, slow the first time)
    oisst/oisst_YYYY.nc NOAA OISST v2.1, one file per year 1982-2021
                        (downloaded from NOAA ERDDAP if missing)
    cufes_cc.csv        CalCOFI CUFES egg samples (downloaded from ERDDAP if missing)

Outputs (in figures/rq3/):
    figB1_exposure_by_period.png   report Figure 3.2
    figB2_where_fish_spawned.png   report Figure 3.3
    figB3b_model_sets.png          report Figure 3.5
    table_*.csv                    every number quoted in the RQ3 results
    rq3_headline_numbers.csv       short list of the key numbers

Everything random uses one seed (SEED = 42), so a re-run gives the same numbers.
The full run takes about 10-15 minutes on a laptop (mostly the model fitting).
"""

import os
import urllib.parse
import urllib.request
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")                      # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN land cells in OISST
warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================
# 1. SETTINGS
# =====================================================================
HERE = os.path.dirname(os.path.abspath(__file__))     # paths work from any working folder
DATA_DIR = os.path.join(HERE, "data")
GDP_FILE = os.path.join(DATA_DIR, "gdp_cc_subset.pkl")
OISST_DIR = os.path.join(DATA_DIR, "oisst")
CUFES_FILE = os.path.join(DATA_DIR, "cufes_cc.csv")
FIG_DIR = os.path.join(HERE, "figures", "rq3")
os.makedirs(FIG_DIR, exist_ok=True)

LON = (-135.0, -115.0)                     # study box
LAT = (25.0, 48.0)
T0, T1 = pd.Timestamp("2012-01-01"), pd.Timestamp("2021-12-31")   # study years
BASE = (1982, 2011)                        # 30-year OISST baseline, both events outside it

QUALITY_FLAG = 5                           # GDP flg_sst value kept as high quality
MIN_DAYS = 5                               # heatwave = 5+ days in a row above the threshold
MIN_HOURS_PER_DAY = 12                     # hourly values needed to make one drifter-day
WIN = 30                                   # "recent exposure" window, days before each sample
BOOT = 1000                                # bootstrap resamples for shares
REPS_AUC = 300                             # bootstrap resamples for model AUCs
SEED = 42
rng = np.random.default_rng(SEED)          # ONE random generator, used in a fixed order

# Preferred surface temperature range of each species (deg C)
SPECIES = {
    "sardine": dict(name="Pacific sardine", col="sardine_eggs", lo=12.0, hi=17.0),
    "anchovy": dict(name="Northern anchovy", col="anchovy_eggs", lo=11.5, hi=16.5),
}

BLOB_COLOUR, MHW_COLOUR = "#d55e00", "#0072b2"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "font.size": 9.5, "legend.frameon": False})

# Rough US West Coast outline (lat, lon), only for drawing the maps
COAST = np.array([[32.53, -117.12], [33.00, -117.28], [33.45, -117.70], [33.75, -118.40], [34.03, -118.55],
                  [34.40, -119.50], [34.46, -120.47], [34.90, -120.65], [35.40, -120.90], [35.67, -121.30],
                  [36.30, -121.90], [36.60, -121.95], [36.95, -122.05], [37.50, -122.50], [37.80, -122.50],
                  [38.30, -123.05], [38.90, -123.70], [39.50, -123.80], [40.40, -124.40], [41.00, -124.10],
                  [41.75, -124.20], [42.80, -124.55], [43.40, -124.30], [44.60, -124.07], [45.50, -123.97],
                  [46.20, -124.05], [46.90, -124.10], [47.60, -124.40], [48.38, -124.70]])

HEADLINE = {}                              # key numbers, saved at the end


def save_fig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  saved figure", name + ".png")


def save_table(df, name, index=True):
    df.to_csv(os.path.join(FIG_DIR, name + ".csv"), index=index)
    print("  saved table ", name + ".csv")


def pct_ci(est):
    """Format (estimate, low, high) shares as '44% [33%-55%]'."""
    p, lo, hi = est
    if not np.isfinite(lo):
        return f"{100 * p:.0f}%"
    return f"{100 * p:.0f}% [{100 * lo:.0f}%-{100 * hi:.0f}%]"


def distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (haversine) in km."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlon = np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# =====================================================================
# 2. LOAD DATA (each loader downloads the file once if it is missing)
# =====================================================================
def load_drifters():
    """GDP hourly drifters in the study box, one row per hourly observation."""
    if not os.path.exists(GDP_FILE):
        print("Downloading GDP hourly data with clouddrift (slow, ~20 GB index the first time) ...")
        import clouddrift.ragged as ragged
        from clouddrift.datasets import gdp1h
        ds = gdp1h()
        sub = ragged.subset(ds, {"lat": LAT, "lon": LON, "time": (np.datetime64(T0), np.datetime64(T1 + pd.Timedelta(hours=23)))},
                            row_dim_name="traj", obs_dim_name="obs", full_rows=False)
        rows = sub["rowsize"].values
        df = pd.DataFrame({"id": np.repeat(sub["id"].values, rows),            # one id per hourly row
                           "time": sub["time"].values, "lat": sub["lat"].values, "lon": sub["lon"].values,
                           "sst": sub["sst"].values, "flg_sst": sub["flg_sst"].values})
        if df.sst.median() > 100:                                               # Kelvin -> Celsius
            df["sst"] -= 273.15
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_pickle(GDP_FILE)
    df = pd.read_pickle(GDP_FILE)
    df["sst_ok"] = np.isfinite(df.sst) & (df.flg_sst == QUALITY_FLAG)
    return df


ERDDAP_OISST = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.nc"


def oisst_url(year):
    """ERDDAP query for one year of daily OISST over the study box (plus a small margin)."""
    q = (f"sst[({year}-01-01T12:00:00Z):1:({year}-12-31T12:00:00Z)][(0.0):1:(0.0)]"
         f"[({LAT[0] - 0.375}):1:({LAT[1] + 0.375})][({LON[0] - 0.375}):1:({LON[1] + 0.375})]")
    return ERDDAP_OISST + "?" + urllib.parse.quote(q, safe=":(),.-")


def tidy_oisst(da):
    """Standard names (lat, lon), drop the depth axis, longitudes -180..180, crop to the box."""
    da = da.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in da.coords})
    if "zlev" in da.dims:
        da = da.isel(zlev=0, drop=True)
    da = da.assign_coords(lon=((da["lon"] + 180) % 360) - 180).sortby("lon").sortby("lat")
    return da.sel(lat=slice(LAT[0] - 0.3, LAT[1] + 0.3), lon=slice(LON[0] - 0.3, LON[1] + 0.3))


def load_oisst():
    """Daily OISST, 1982-2021, as one xarray DataArray (time, lat, lon)."""
    os.makedirs(OISST_DIR, exist_ok=True)
    pieces = []
    for year in range(BASE[0], T1.year + 1):
        f = os.path.join(OISST_DIR, f"oisst_{year}.nc")
        if not os.path.exists(f):
            print(f"  downloading OISST {year} ...")
            tmp = f + ".part"
            urllib.request.urlretrieve(oisst_url(year), tmp)
            with xr.open_dataset(tmp) as ds:
                tidy_oisst(ds["sst"]).load().to_netcdf(f)
            os.remove(tmp)
        pieces.append(xr.open_dataarray(f).load().astype("float32"))
    da = xr.concat(pieces, dim="time").sortby("time")
    return da.assign_coords(time=pd.DatetimeIndex(da.time.values).floor("D"))   # one value per calendar day


def load_eggs():
    """CalCOFI CUFES egg samples (ship SST + egg counts), downloaded from ERDDAP if missing."""
    if not os.path.exists(CUFES_FILE):
        print("  downloading CalCOFI CUFES egg samples ...")
        cols = "time,latitude,longitude,start_temperature,sardine_eggs,anchovy_eggs,hake_eggs,jack_mackerel_eggs,squid_eggs"
        q = (cols + "&time>=2012-01-01T00:00:00Z&time<=2021-12-31T23:59:59Z"
             "&latitude>=25&latitude<=48&longitude>=-135&longitude<=-115")
        url = "https://coastwatch.pfeg.noaa.gov/erddap/tabledap/erdCalCOFIcufes.csv?" + urllib.parse.quote(q, safe="=,&:")
        pd.read_csv(url, skiprows=[1]).to_csv(CUFES_FILE, index=False)        # row 2 of ERDDAP csv = units
    cu = pd.read_csv(CUFES_FILE)
    cu = cu[pd.to_numeric(cu.latitude, errors="coerce").notna()].copy()      # drop any units row
    for c in ["latitude", "longitude", "start_temperature", "sardine_eggs", "anchovy_eggs"]:
        cu[c] = pd.to_numeric(cu[c], errors="coerce")
    cu["time"] = pd.to_datetime(cu.time, utc=True).dt.tz_localize(None)
    cu["date"] = cu.time.dt.floor("D")
    cu["y"], cu["m"], cu["doy"] = cu.time.dt.year, cu.time.dt.month, cu.time.dt.day_of_year
    for sp, v in SPECIES.items():
        cu[sp] = cu[v["col"]] > 0                                            # True = eggs present
    return cu


def period_of(t):
    """Label each date with its period. The number in front keeps the periods in time order."""
    if t < pd.Timestamp("2014-01-01"):
        return "1 Before (2012-13)"
    if t < pd.Timestamp("2017-01-01"):
        return "2 Blob (2014-16)"
    if t < pd.Timestamp("2019-01-01"):
        return "3 Between (2017-18)"
    if t < pd.Timestamp("2021-01-01"):
        return "4 MHW (2019-20)"
    return "5 After (2021)"


# =====================================================================
# 3. HEATWAVE THRESHOLD AND FLAGS (Hobday et al. 2016)
# =====================================================================
def hobday_threshold(da, base=BASE, pct=90, half_win=5, smooth=31):
    """
    90th percentile of SST for each day of the year and each grid cell, from the baseline years.
    Uses all days within +-5 days of that calendar day, then a 31-day moving average.
    Returns an array (366, lat, lon).
    """
    b = da.sel(time=slice(f"{base[0]}-01-01", f"{base[1]}-12-31"))
    arr = b.values
    doy = pd.DatetimeIndex(b.time.values).day_of_year.values
    thr = np.full((366,) + arr.shape[1:], np.nan, dtype=np.float32)
    for d in range(1, 367):
        gap = (doy - d + 183) % 366 - 183                  # days between each baseline day and day d (wraps at new year)
        thr[d - 1] = np.nanpercentile(arr[np.abs(gap) <= half_win], pct, axis=0)

    # 31-day circular moving average that ignores NaN (land cells stay NaN)
    k = smooth // 2
    padded = np.concatenate([thr[-k:], thr, thr[:k]], axis=0)      # wrap Dec <-> Jan
    total = np.cumsum(np.where(np.isnan(padded), 0, padded), axis=0, dtype=np.float64)
    count = np.cumsum(~np.isnan(padded), axis=0)
    total = np.concatenate([np.zeros((1,) + total.shape[1:]), total])
    count = np.concatenate([np.zeros((1,) + count.shape[1:], int), count])
    n = count[smooth:] - count[:-smooth]
    out = (total[smooth:] - total[:-smooth]) / np.maximum(n, 1)
    out[n == 0] = np.nan
    return out.astype(np.float32)


def heatwave_flag(above, n=MIN_DAYS):
    """True on each day that is part of a run of n or more days in a row above the threshold."""
    fwd = np.zeros(above.shape, np.int16)      # run length counted forwards in time
    bwd = np.zeros(above.shape, np.int16)      # run length counted backwards in time
    run = np.zeros(above.shape[1:], np.int16)
    for t in range(above.shape[0]):
        run = np.where(above[t], run + 1, 0).astype(np.int16)
        fwd[t] = run
    run = np.zeros(above.shape[1:], np.int16)
    for t in range(above.shape[0] - 1, -1, -1):
        run = np.where(above[t], run + 1, 0).astype(np.int16)
        bwd[t] = run
    return above & ((fwd + bwd - 1) >= n)      # full run length through each day


# =====================================================================
# MAIN ANALYSIS
# =====================================================================
print("1) Loading data")
df = load_drifters()
print(f"  {len(df):,} hourly drifter observations from {df.id.nunique()} drifters")

oisst = load_oisst()
thr = hobday_threshold(oisst)                                  # (366, lat, lon)
study = oisst.sel(time=slice(T0, T1))                          # 2012-2021 only
TAX = pd.DatetimeIndex(study.time.values)                      # the study days
SST = study.values                                             # (day, lat, lon)
lat_g, lon_g = study.lat.values, study.lon.values
above_p90 = SST > thr[TAX.day_of_year.values - 1]                # above the threshold that day
MHW = heatwave_flag(above_p90)                                 # in a 5+ day heatwave that day
print(f"  OISST {oisst.shape}, baseline {BASE[0]}-{BASE[1]}")


def nearest_cell(lat, lon):
    """Index of the nearest OISST grid row (lat) and column (lon)."""
    iy = np.abs(np.asarray(lat)[:, None] - lat_g[None, :]).argmin(1)
    ix = np.abs(np.asarray(lon)[:, None] - lon_g[None, :]).argmin(1)
    return iy, ix


# ---- drifter-days: daily mean SST per drifter, matched to OISST ----
good = df[df.sst_ok].assign(date=lambda x: x.time.dt.floor("D"))
daily = (good.groupby(["id", "date"])
             .agg(sst=("sst", "mean"), lat=("lat", "mean"), lon=("lon", "mean"), n=("sst", "size"))
             .reset_index())
daily = daily[daily.n >= MIN_HOURS_PER_DAY].copy()              # need 12+ hours for a fair daily mean
daily["iy"], daily["ix"] = nearest_cell(daily.lat.values, daily.lon.values)
daily["ti"] = TAX.get_indexer(daily.date)                      # -1 = day not in OISST
daily = daily[daily.ti >= 0].copy()
daily["thr"] = thr[daily.date.dt.day_of_year.values - 1, daily.iy.values, daily.ix.values]
daily["oisst_sst"] = SST[daily.ti.values, daily.iy.values, daily.ix.values]
daily = daily[np.isfinite(daily.thr) & np.isfinite(daily.oisst_sst)].reset_index(drop=True)
daily["oisst_mhw"] = MHW[daily.ti.values, daily.iy.values, daily.ix.values]
print(f"  {len(daily):,} drifter-days from {daily.id.nunique()} drifters")

# ---- egg samples: match each sample to its OISST cell and day ----
cu = load_eggs()
cu["period"] = cu.time.map(period_of)
cu["iy"], cu["ix"] = nearest_cell(cu.latitude.values, cu.longitude.values)
cu["ti"] = TAX.get_indexer(cu.date)
cu = cu[(cu.ti >= 0) & np.isfinite(SST[cu.ti.values.clip(0), cu.iy.values, cu.ix.values])].reset_index(drop=True)
ti, iy, ix = cu.ti.values, cu.iy.values, cu.ix.values
cu["oisst_sst"] = SST[ti, iy, ix]
cu["anom_p90"] = cu.oisst_sst - thr[TAX.day_of_year.values[ti] - 1, iy, ix]   # deg C above (+) or below (-) the threshold
cu["in_mhw"] = MHW[ti, iy, ix]


def recent_share(flags, win=WIN):
    """Share of the last `win` days (today included) that were True at each sample's grid cell."""
    days_back = ti[:, None] - np.arange(win)[None, :]
    ok = days_back >= 0
    vals = np.where(ok, flags[np.clip(days_back, 0, None), iy[:, None], ix[:, None]], False)
    return vals.sum(1) / np.maximum(ok.sum(1), 1)


cu["mhw_recent"] = recent_share(MHW)
for sp, v in SPECIES.items():
    cu[f"above_{sp}"] = cu.oisst_sst > v["hi"]                         # above the species' upper limit today
    cu[f"above_{sp}_recent"] = recent_share(SST > v["hi"])             # share of last 30 days above it

spring = cu[cu.m.isin([3, 4, 5])].copy()                               # spawning season
core = spring[spring.latitude.between(31.5, 35.5)].copy()              # surveyed every spring
print(f"  {len(cu):,} egg samples matched; spring {len(spring):,}; spring core area {len(core):,}")


def boot_share(g, col, B=BOOT):
    """Mean of a True/False column, with a 95% interval from resampling whole survey DAYS
    (samples taken on the same day are not independent)."""
    days = g.date.values
    unique_days = np.unique(days)
    rows_of = {d: np.flatnonzero(days == d) for d in unique_days}
    v = g[col].values.astype(float)
    if len(unique_days) < 3:
        return v.mean(), np.nan, np.nan
    boots = [v[np.concatenate([rows_of[d] for d in rng.choice(unique_days, len(unique_days))])].mean()
             for _ in range(B)]
    return v.mean(), *np.percentile(boots, [2.5, 97.5])


# =====================================================================
# B0. Do the temperature inputs match the ship temperature at the eggs?
# =====================================================================
print("\nB0) Temperature inputs vs ship SST")
pairs = []
for shift in (-1, 0, 1):                                               # same day, or one day either side
    q = (cu[["date", "latitude", "longitude", "start_temperature"]]
         .assign(date=lambda x: x.date + pd.Timedelta(days=shift)).reset_index())
    m = q.merge(daily[["id", "date", "sst", "lat", "lon"]], on="date")
    m["km"] = distance_km(m.latitude.values, m.longitude.values, m.lat.values, m.lon.values)
    pairs.append(m[m.km <= 50])                                        # within 50 km
pairs = pd.concat(pairs).sort_values("km").drop_duplicates("index")    # keep the closest drifter per sample
drifter_minus_ship = pairs.sst - pairs.start_temperature
oisst_minus_ship = cu.oisst_sst - cu.start_temperature
B0 = pd.DataFrame({
    "pairs": [len(pairs), len(cu)],
    "mean difference (°C)": [drifter_minus_ship.mean(), oisst_minus_ship.mean()],
    "SD (°C)": [drifter_minus_ship.std(), oisst_minus_ship.std()],
    "correlation r": [np.corrcoef(pairs.sst, pairs.start_temperature)[0, 1],
                      np.corrcoef(cu.oisst_sst, cu.start_temperature)[0, 1]]},
    index=["drifter vs ship (within 50 km, 1 day)", "OISST vs ship (same cell, same day)"]).round(2)
print(B0.to_string())
save_table(B0, "table_B0_temperature_inputs")
HEADLINE["B0 drifter-ship pairs"] = len(pairs)
HEADLINE["B0 drifter minus ship (C)"] = round(float(drifter_minus_ship.mean()), 2)
HEADLINE["B0 drifter-ship r"] = round(float(B0.iloc[0]["correlation r"]), 2)
HEADLINE["B0 OISST-ship SD (C)"] = round(float(B0.iloc[1]["SD (°C)"]), 2)

# =====================================================================
# B1. How much heatwave exposure reached the spawning area in each period?
# =====================================================================
print("\nB1) Exposure of the spawning area (spring, 31.5-35.5N)")
rows = []
for p, g in core.groupby("period"):
    r = {"period": p[2:], "samples": len(g), "survey days": g.date.nunique(),
         "in heatwave that day": pct_ci(boot_share(g, "in_mhw")),
         "heatwave days, previous 30": pct_ci(boot_share(g, "mhw_recent")),
         "median °C vs P90": round(float(g.anom_p90.median()), 2)}
    for sp, v in SPECIES.items():
        r[f"above {v['name']} limit ({v['hi']:g}°C)"] = pct_ci(boot_share(g, f"above_{sp}"))
    rows.append(r)
B1 = pd.DataFrame(rows).set_index("period")
print(B1.to_string())
save_table(B1, "table_B1_exposure_core_spring")
HEADLINE["B1 in heatwave, Blob"] = B1.loc["Blob (2014-16)", "in heatwave that day"]
HEADLINE["B1 in heatwave, 2019-20"] = B1.loc["MHW (2019-20)", "in heatwave that day"]

# Cross-check with drifters: same season and latitudes, coastal side (east of 125W)
dsp = daily[daily.date.dt.month.isin([3, 4, 5]) & daily.lat.between(31.5, 35.5) & (daily.lon >= -125)].copy()
dsp["period"] = dsp.date.map(period_of)
dsp["above_sardine"] = dsp.sst > SPECIES["sardine"]["hi"]
dsp["above_anchovy"] = dsp.sst > SPECIES["anchovy"]["hi"]
B1d = dsp.groupby("period").agg(drifter_days=("sst", "size"), drifters=("id", "nunique"),
                                in_heatwave_pct=("oisst_mhw", lambda s: 100 * s.mean()),
                                above_sardine_pct=("above_sardine", lambda s: 100 * s.mean()),
                                above_anchovy_pct=("above_anchovy", lambda s: 100 * s.mean()))
B1d.index = [i[2:] for i in B1d.index]
print(B1d.round(1).to_string())
save_table(B1d, "table_B1_exposure_drifters")

# Figure B1 (report Figure 3.2): grouped bars with 95% intervals
order = sorted(core.period.unique())
bars = {"in_mhw": ("in a heatwave that day", "#d55e00"),
        "above_sardine": ("above sardine limit (17°C)", "#0072b2"),
        "above_anchovy": ("above anchovy limit (16.5°C)", "#e69f00")}
fig, ax = plt.subplots(figsize=(10, 4))
w = 0.26
for k, (col, (label, colour)) in enumerate(bars.items()):
    est = np.array([boot_share(core[core.period == o], col) for o in order]) * 100   # rows: (mean, lo, hi)
    x = np.arange(len(order)) + (k - 1) * w
    ax.bar(x, est[:, 0], w, color=colour, label=label)
    ax.errorbar(x, est[:, 0], yerr=[est[:, 0] - est[:, 1], est[:, 2] - est[:, 0]],
                fmt="none", ecolor="k", lw=0.8, capsize=2)
ax.set_xticks(np.arange(len(order)))
ax.set_xticklabels([o[2:].replace(" (", "\n(") for o in order])
ax.set_ylabel("% of spring survey samples, 31.5–35.5°N")
ax.set_ylim(0, 112)
ax.legend(fontsize=8.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13))
ax.set_title("Thermal exposure of spawning habitat (same place, same season; 95% CI over survey days)", pad=34)
fig.tight_layout()
save_fig(fig, "figB1_exposure_by_period")

# =====================================================================
# B2. Did the place where fish spawned change?
# =====================================================================
print("\nB2) Egg presence and spawning latitude")
rows = []
for p, g in core.groupby("period"):
    rows.append({"period": p[2:], "samples": len(g), "ship SST (median °C)": g.start_temperature.median(),
                 **{f"{v['name']} eggs present": pct_ci(boot_share(g, sp)) for sp, v in SPECIES.items()}})
B2a = pd.DataFrame(rows).set_index("period")
print(B2a.round(1).to_string())
save_table(B2a, "table_B2_eggs_core_spring")

rows = []
for sp, v in SPECIES.items():
    with_eggs = spring[spring[sp]]
    for p, g in with_eggs.groupby("period"):
        rows.append({"species": v["name"], "period": p[2:], "samples with eggs": len(g),
                     "median latitude": g.latitude.median(),
                     "egg-weighted latitude": np.average(g.latitude, weights=g[v["col"]]),
                     "ship SST at eggs (median)": g.start_temperature.median()})
B2b = pd.DataFrame(rows).round(2)
print(B2b.to_string(index=False))
save_table(B2b, "table_B2_spawning_latitude", index=False)

# Figure B2 (report Figure 3.3): two maps + yearly egg presence in the core area
fig = plt.figure(figsize=(14, 5.2))
grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])
maps = [("Spring 2012 (before the Blob)", spring[spring.y == 2012]),
        ("Spring 2015–16 (Blob peak)", spring[spring.y.isin([2015, 2016])])]
for k, (title, g) in enumerate(maps):
    ax = fig.add_subplot(grid[k])
    ax.plot(COAST[:, 1], COAST[:, 0], color="#777", lw=0.9)
    ax.scatter(g.longitude, g.latitude, s=3, color="#d0d0d0", label="surveyed, no sardine eggs")
    eggs = g[g.sardine]
    ax.scatter(eggs.longitude, eggs.latitude, s=8 + 3 * np.sqrt(eggs.sardine_eggs), color="#0072b2",
               alpha=0.75, edgecolor="k", lw=0.2, label="sardine eggs (size = count)")
    ax.set_xlim(-129.5, -116.5)
    ax.set_ylim(29.5, 48.2)
    ax.set_aspect(1 / np.cos(np.deg2rad(38)))          # so degrees look the right shape at 38N
    ax.set_title(f"{title}\nmedian latitude of eggs {eggs.latitude.median():.1f}°N, "
                 f"at {eggs.start_temperature.median():.1f}°C", fontsize=10)
    ax.set_xlabel("longitude")
    if k == 0:
        ax.set_ylabel("latitude")
        ax.legend(fontsize=7.5, loc="lower left")

ax = fig.add_subplot(grid[2])
yearly = core.groupby("y").agg(sard=("sardine", "mean"), anch=("anchovy", "mean"), sst=("start_temperature", "median"))
ax.axvspan(2013.5, 2016.5, color=BLOB_COLOUR, alpha=0.13, lw=0)
ax.axvspan(2018.5, 2020.5, color=MHW_COLOUR, alpha=0.13, lw=0)
ax.plot(yearly.index, 100 * yearly.sard, "o-", color="#0072b2", lw=2, label="sardine eggs present (%)")
ax.plot(yearly.index, 100 * yearly.anch, "s-", color="#e69f00", lw=2, label="anchovy eggs present (%)")
ax.set_ylabel("% of spring samples, 31.5–35.5°N")
ax.set_xticks(yearly.index)
ax.legend(fontsize=8, loc="upper left")
ax2 = ax.twinx()                                        # second y-axis for temperature
ax2.plot(yearly.index, yearly.sst, "--", color="#d55e00", lw=1.4)
ax2.set_ylabel("ship SST, median (°C, dashed)", color="#d55e00")
ax2.grid(False)
ax2.spines["right"].set_visible(True)
ax.set_title("Southern/central California, spring only (same place, same season)", fontsize=10)
fig.suptitle("Where did fish spawn? CalCOFI egg surveys during the Blob (shaded orange) and 2019–20 (shaded blue)",
             y=1.02, fontsize=12)
fig.tight_layout()
save_fig(fig, "figB2_where_fish_spawned")

# =====================================================================
# B3. Does heatwave exposure help predict where eggs were found?
#     Three nested inputs:  A = place + season
#                           B = A + temperature range
#                           C = B + heatwave exposure
# =====================================================================
print("\nB3) Logistic regression, leave-one-year-out")
_above_cache = {}


def above_recent_shifted(d, hi):
    """Share of the last WIN days above a shifted upper limit `hi` (only used in the B4 sensitivity test)."""
    key = (hi, WIN)
    if key not in _above_cache:
        _above_cache[key] = pd.Series(recent_share(SST > hi), index=cu.index)
    return _above_cache[key].loc[d.index].values


def design(d, sp, kind, shift=0.0):
    """Build the input matrix X for input set A, B or C."""
    v = SPECIES[sp]
    lo_t, hi_t = v["lo"] + shift, v["hi"] + shift
    la = (d.latitude.values - 38.0) / 5.0                       # centred and scaled position
    lo = (d.longitude.values + 122.0) / 4.0
    season = 2 * np.pi * d.doy.values / 365.25                  # day of year as an angle
    X = [la, la ** 2, lo, lo ** 2, la * lo, np.sin(season), np.cos(season)]      # A: place + season
    if kind in ("B", "C"):
        s = d.oisst_sst.values
        X += [((s >= lo_t) & (s <= hi_t)).astype(float),        # B: inside the preferred range
              np.clip(s - hi_t, 0, None),                       #    degrees above the range
              np.clip(lo_t - s, 0, None)]                       #    degrees below the range
    if kind == "C":
        above_recent = d[f"above_{sp}_recent"].values if shift == 0 else above_recent_shifted(d, hi_t)
        X += [d.in_mhw.values.astype(float),                    # C: in a heatwave today
              d.mhw_recent.values,                              #    share of last 30 days in a heatwave
              above_recent]                                     #    share of last 30 days above the range
    return np.column_stack(X)


def leave_one_year_out(d, sp, kind, make_model, shift=0.0):
    """Fit on all years but one, predict the held-out year, repeat for every year."""
    X, y, years = design(d, sp, kind, shift), d[sp].values.astype(int), d.y.values
    p = np.full(len(d), np.nan)
    for yr in np.unique(years):
        train, test = years != yr, years == yr
        if y[train].min() == y[train].max():                    # need both classes to fit
            continue
        model = make_model().fit(X[train], y[train])
        p[test] = model.predict_proba(X[test])[:, 1]
    return p


def logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def boosting():
    # shallow, slow-learning trees with a minimum leaf size to limit overfitting
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=SEED)


def loyo(d, sp, kind, shift=0.0):
    return leave_one_year_out(d, sp, kind, logistic, shift)


# Bootstrap over survey days, shared by every model so the gains are paired
days = cu.date.values
rows_of_day = {d: np.flatnonzero(days == d) for d in np.unique(days)}
unique_days = np.unique(days)
REPS = [np.concatenate([rows_of_day[d] for d in rng.choice(unique_days, len(unique_days))]) for _ in range(REPS_AUC)]


def auc_ci(y, p):
    """ROC-AUC, its 95% interval, and all bootstrap AUCs (for paired differences)."""
    b = np.array([roc_auc_score(y[r], p[r]) for r in REPS if y[r].min() != y[r].max()])
    return roc_auc_score(y, p), *np.percentile(b, [2.5, 97.5]), b


PRED, rows, gains = {}, [], []
for sp, v in SPECIES.items():
    y = cu[sp].values.astype(int)
    boot = {}
    for kind in "ABC":
        p = loyo(cu, sp, kind)
        PRED[(sp, kind)] = p
        a, lo_, hi_, boot[kind] = auc_ci(y, p)
        rows.append({"species": v["name"], "inputs": kind, "ROC-AUC [95% CI]": f"{a:.3f} [{lo_:.3f}-{hi_:.3f}]",
                     "PR-AUC": average_precision_score(y, p), "Brier": brier_score_loss(y, p), "egg prevalence": y.mean()})
    for k1, k0 in (("B", "A"), ("C", "B"), ("C", "A")):
        lo_, hi_ = np.percentile(boot[k1] - boot[k0], [2.5, 97.5])
        gains.append({"species": v["name"], "comparison": f"{k1} minus {k0}",
                      "AUC gain": roc_auc_score(y, PRED[(sp, k1)]) - roc_auc_score(y, PRED[(sp, k0)]),
                      "95% CI": f"[{lo_:+.3f}, {hi_:+.3f}]", "clear gain": bool(lo_ > 0)})
    HEADLINE[f"B3 {v['name']} logistic AUC A/B/C"] = " / ".join(
        f"{roc_auc_score(y, PRED[(sp, k)]):.3f}" for k in "ABC")
B3, B3g = pd.DataFrame(rows), pd.DataFrame(gains)
print(B3.round(3).to_string(index=False))
print(B3g.round(3).to_string(index=False))
save_table(B3, "table_B3_models", index=False)
save_table(B3g, "table_B3_model_gains", index=False)

# =====================================================================
# B3b. Same inputs, three sets of models: rule-based, logistic, gradient boosting
# =====================================================================
print("\nB3b) Rule-based vs logistic regression vs gradient boosting (slowest step)")
LEVELS = {"A": "A · place + season", "B": "B · + temperature range", "C": "C · + heatwave exposure"}
SETS = ["Rule-based overlay", "Logistic regression", "Gradient boosting"]     # simplest first


def rule_scores(d, sp):
    """No fitted model: the usual egg rate for that 1-degree latitude band and month in the OTHER years,
    then fixed cuts for water outside the range (B) and recent heatwave days (C)."""
    v = SPECIES[sp]
    y, years = d[sp].values.astype(float), d.y.values
    key = np.floor(d.latitude.values).astype(int) * 100 + d.m.values     # band x month
    A = np.full(len(d), np.nan)
    for yr in np.unique(years):
        train, test = years != yr, years == yr
        tab = pd.DataFrame({"k": key[train], "y": y[train]}).groupby("k").y.agg(["sum", "size"])
        p0 = y[train].mean()
        rate = (tab["sum"] + 2 * p0) / (tab["size"] + 2)                 # smoothed toward the overall rate
        A[test] = pd.Series(key[test]).map(rate).fillna(p0).values
    s = d.oisst_sst.values
    inside = ((s >= v["lo"]) & (s <= v["hi"])).astype(float)
    B = A * (0.5 + 0.5 * inside)                     # halve the rate when outside the range
    C = B * (1.0 - 0.5 * d.mhw_recent.values)        # cut further by recent heatwave days
    return {"A": A, "B": B, "C": C}


ALL = {}
for sp in SPECIES:
    rule = rule_scores(cu, sp)
    for k in "ABC":
        ALL[(sp, "Rule-based overlay", k)] = rule[k]
        ALL[(sp, "Logistic regression", k)] = PRED[(sp, k)]
        ALL[(sp, "Gradient boosting", k)] = leave_one_year_out(cu, sp, k, boosting)

rows, BOOTS = [], {}
for sp, v in SPECIES.items():
    y = cu[sp].values.astype(int)
    for st in SETS:
        for k in "ABC":
            p = ALL[(sp, st, k)]
            a, lo_, hi_, BOOTS[(sp, st, k)] = auc_ci(y, p)
            is_prob = (st != "Rule-based overlay") or (k == "A")         # Brier only for real probabilities
            rows.append({"species": v["name"], "set": st, "model": LEVELS[k], "ROC-AUC": a, "lo": lo_, "hi": hi_,
                         "PR-AUC": average_precision_score(y, p),
                         "Brier": brier_score_loss(y, p) if is_prob else np.nan})
SETCMP = pd.DataFrame(rows)
print(SETCMP.round(3).to_string(index=False))
save_table(SETCMP.round(4), "table_B3b_model_sets", index=False)

# Does adding heatwave exposure (C minus A) help in every set?  Paired bootstrap.
rows = []
for sp, v in SPECIES.items():
    y = cu[sp].values.astype(int)
    for st in SETS:
        lo_, hi_ = np.percentile(BOOTS[(sp, st, "C")] - BOOTS[(sp, st, "A")], [2.5, 97.5])
        rows.append({"species": v["name"], "set": st,
                     "AUC gain C - A": roc_auc_score(y, ALL[(sp, st, "C")]) - roc_auc_score(y, ALL[(sp, st, "A")]),
                     "95% CI": f"[{lo_:+.3f}, {hi_:+.3f}]", "clear gain": bool(lo_ > 0)})
SETGAIN = pd.DataFrame(rows)
print(SETGAIN.round(3).to_string(index=False))
save_table(SETGAIN, "table_B3b_exposure_gain_by_set", index=False)

# Decision rule: pick the SIMPLEST set whose best model is not clearly worse than the best model overall
rows = []
for sp, v in SPECIES.items():
    q = SETCMP[SETCMP.species == v["name"]]
    top = q.loc[q["ROC-AUC"].idxmax()]
    top_k = top.model[0]                                   # "A", "B" or "C"
    for st in SETS:
        qs = q[q.set == st]
        best = qs.loc[qs["ROC-AUC"].idxmax()]
        lo_, hi_ = np.percentile(BOOTS[(sp, top.set, top_k)] - BOOTS[(sp, st, best.model[0])], [2.5, 97.5])
        rows.append({"species": v["name"], "set": st, "its best model": best.model, "ROC-AUC": best["ROC-AUC"],
                     "gap to overall best": top["ROC-AUC"] - best["ROC-AUC"], "gap 95% CI": f"[{lo_:+.3f}, {hi_:+.3f}]",
                     "not clearly worse": bool(lo_ <= 0)})
SEL = pd.DataFrame(rows)
print(SEL.round(3).to_string(index=False))
save_table(SEL, "table_B3b_set_selection", index=False)
for sp, v in SPECIES.items():
    chosen = SEL[(SEL.species == v["name"]) & SEL["not clearly worse"]].iloc[0]     # first = simplest
    print(f"  {v['name']}: chosen set = {chosen.set} ({chosen['its best model']}, AUC {chosen['ROC-AUC']:.3f})")
    HEADLINE[f"B3b chosen set, {v['name']}"] = f"{chosen.set} ({chosen['its best model']}, AUC {chosen['ROC-AUC']:.3f})"
    HEADLINE[f"B3b gain C-A rule/logistic/boosting, {v['name']}"] = " / ".join(
        f"{r['AUC gain C - A']:+.3f}{'*' if r['clear gain'] else ''}"
        for _, r in SETGAIN[SETGAIN.species == v["name"]].iterrows())

# Figure B3b (report Figure 3.5): AUC of every model, grouped by set, one panel per species
colours = {"A": "#9aa0a6", "B": "#56b4e9", "C": "#d55e00"}
fig, axes = plt.subplots(1, len(SPECIES), figsize=(13, 4.4), sharey=True)
for ax, (sp, v) in zip(axes, SPECIES.items()):
    for i, st in enumerate(SETS):
        for k, kind in enumerate("ABC"):
            r = SETCMP[(SETCMP.species == v["name"]) & (SETCMP.set == st) & (SETCMP.model == LEVELS[kind])].iloc[0]
            ax.errorbar(i + (k - 1) * 0.22, r["ROC-AUC"], yerr=[[r["ROC-AUC"] - r.lo], [r.hi - r["ROC-AUC"]]],
                        fmt="o", color=colours[kind], ms=7, capsize=3, label=LEVELS[kind] if i == 0 else None)
        g = SETGAIN[(SETGAIN.species == v["name"]) & (SETGAIN.set == st)].iloc[0]
        ax.text(i, 0.47, f"C − A: {g['AUC gain C - A']:+.3f}{' *' if g['clear gain'] else ''}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(SETS)))
    ax.set_xticklabels(SETS)
    ax.axhline(0.5, color="#999", ls="--", lw=0.8)               # 0.5 = no better than chance
    ax.set_title(v["name"])
    ax.set_ylim(0.44, 0.9)
axes[0].set_ylabel("ROC-AUC, leave-one-year-out (95% CI)")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Three sets of models, same inputs: does adding heatwave exposure help whichever model is used?  "
             "(* = gain interval above 0)", y=1.02)
fig.tight_layout()
save_fig(fig, "figB3b_model_sets")

# =====================================================================
# B4. Sensitivity: does the B3 answer hold for other ranges and windows?
# =====================================================================
print("\nB4) Sensitivity of the logistic C - A gain")
rows = []
for sp, v in SPECIES.items():
    y = cu[sp].values.astype(int)
    auc_a = roc_auc_score(y, PRED[(sp, "A")])                    # A has no temperature input, so it does not change
    for shift in (-2, -1, 0, 1, 2):                              # move the whole range by this many deg C
        rows.append({"species": v["name"], "range shift (°C)": shift, "range": f"{v['lo'] + shift:g}-{v['hi'] + shift:g}",
                     "AUC A": auc_a, "AUC B": roc_auc_score(y, loyo(cu, sp, "B", shift)),
                     "AUC C": roc_auc_score(y, loyo(cu, sp, "C", shift))})
B4 = pd.DataFrame(rows)
B4["C minus A"] = B4["AUC C"] - B4["AUC A"]
print(B4.round(3).to_string(index=False))
save_table(B4, "table_B4_sensitivity_range", index=False)

rows = []
for win in (10, 30, 60):                                         # recent-exposure window in days
    test = cu.copy()
    test["mhw_recent"] = recent_share(MHW, win)
    for sp, v in SPECIES.items():
        test[f"above_{sp}_recent"] = recent_share(SST > v["hi"], win)
        y = test[sp].values.astype(int)
        rows.append({"species": v["name"], "exposure window (d)": win,
                     "AUC C": roc_auc_score(y, loyo(test, sp, "C")), "AUC A": roc_auc_score(y, PRED[(sp, "A")])})
B4w = pd.DataFrame(rows)
B4w["C minus A"] = B4w["AUC C"] - B4w["AUC A"]
print(B4w.round(3).to_string(index=False))
save_table(B4w, "table_B4_sensitivity_window", index=False)
for v in SPECIES.values():
    g1, g2 = B4[B4.species == v["name"]]["C minus A"], B4w[B4w.species == v["name"]]["C minus A"]
    HEADLINE[f"B4 {v['name']} gain range (ranges / windows)"] = (f"{g1.min():+.3f} to {g1.max():+.3f} / "
                                                                f"{g2.min():+.3f} to {g2.max():+.3f}")

# =====================================================================
# Save the headline numbers
# =====================================================================
summary = pd.Series(HEADLINE, name="value").to_frame()
save_table(summary, "rq3_headline_numbers")
print("\nHeadline numbers:")
print(summary.to_string())
print("\nDone. All outputs are in", FIG_DIR)
