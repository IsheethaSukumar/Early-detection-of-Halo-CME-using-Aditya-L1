# 📖 Notebook Explanation — SWIS & CME Preprocessing Pipeline

---

## 🗺️ Big Picture

The notebook's job is to answer: **"Given solar wind measurements, can we predict whether a CME (Coronal Mass Ejection) is occurring?"**

It does this by:
1. Loading raw data from two sources (SWIS `.cdf` sensor files + CME event catalogues)
2. Putting them on the **same hourly time axis**
3. Creating features and labels
4. Splitting and exporting training-ready arrays

---

## Section 0 — Install & Import Dependencies

```python
REQUIRED = ['cdflib', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn', 'tqdm']

for pkg in REQUIRED:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
```

### What it does
- Loops through every required library
- **Tries to import it** — if it's missing (`ImportError`), installs it automatically
- You never have to manually pip-install anything before running

### Why `cdflib`?
NASA stores satellite data in **CDF (Common Data Format)** — a binary format for time-series sensor data. `cdflib` is the Python library that reads `.cdf` files.

---

## Section 1 — Load & Inspect CSV Sources

### 1a. Solar Wind CSV

```python
sw_df = pd.read_csv(SW_CSV_PATH, parse_dates=['timestamp'])
sw_df.set_index('timestamp', inplace=True)
sw_df.sort_index(inplace=True)
sw_df.index = sw_df.index.tz_localize(None)
```

| Column | Meaning |
|--------|---------|
| `bx_gse, by_gse, bz_gse` | Magnetic field components in GSE coordinates (nanoTesla) |
| `by_gsm, bz_gsm` | Magnetic field in GSM coordinates |
| `proton_density` | Number of protons per cm³ |
| `plasma_speed_kmps` | Solar wind bulk speed (km/s) |
| `flow_pressure_npa` | Dynamic ram pressure (nanoPascal) |
| `plasma_temperature_k` | Proton temperature (Kelvin) |

**Why `tz_localize(None)`?** 
- Some CSVs have timezone-aware timestamps (e.g. `2021-01-01 00:00:00+00:00`)
- The SWIS data is timezone-naive
- We strip timezone info so both can be joined on the same index

### 1b. CME Labels

```python
def load_cme(path):
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'date' in df.columns and 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(
            df['date'].astype(str) + ' ' + df['time'].astype(str), ...)
```

**Why the if/elif?**
- The two CME files have slightly different formats:
  - `cme_labels_2021_2024.csv` → already has a `timestamp` column
  - `cme_labels_2024_2025.csv` → has separate `date` and `time` columns
- The function handles both automatically

| Column | Meaning |
|--------|---------|
| `central_pa` | Central position angle of the CME (degrees) |
| `angular_width` | How wide the CME is (degrees; 360° = halo CME) |
| `speed_kmps` | CME speed (km/s) |
| `is_halo` | 1 if the CME spreads in all directions (most dangerous) |
| `label` | **1 = CME event, 0 = normal solar activity** |

### 1c. Combining Both CME Catalogues

```python
common_cols = list(set(cme_early.columns) & set(cme_late.columns))
cme_all = pd.concat([cme_early[common_cols], cme_late[common_cols]], axis=0)
cme_all = cme_all[~cme_all.index.duplicated(keep='first')]
```

- `set(A) & set(B)` → finds columns present in **both** files (intersection)
- `pd.concat` stacks the two tables vertically (same columns, more rows)
- `.duplicated()` removes any event logged in both catalogues

---

## Section 2 — Load SWIS CDF Files

### Why SWIS files exist
The **SWIS (Solar Wind Ion Spectrometer)** is an instrument on India's **Aditya-L1** spacecraft (launched 2023). It sits at the L1 Lagrange point between Earth and Sun, measuring the solar wind continuously.

Each day produces one `.cdf` file with ~34,000 records at sub-second cadence.

### 2a. Finding Files & Deduplication

```python
DATE_RE = re.compile(r'(\d{8})_UNP.*?(V\d+)\.cdf$')
for fp in all_cdf:
    m = DATE_RE.search(os.path.basename(fp))
    if m:
        date_to_file[m.group(1)].append((m.group(2), fp))

unique_cdf = [sorted(v)[-1][1] for v in date_to_file.values()]
```

**Filename example:** `AL1_ASW91_L1_AUX_20240507_UNP_9999_999999_V02.cdf`

- The regex extracts: date = `20240507`, version = `V02`
- For days with both `V01` and `V02`, we keep only **V02** (latest revision)
- `sorted(v)[-1]` picks the alphabetically last version (highest = newest)

### 2b. Peeking at the CDF Structure

```python
sample_cdf = cdflib.CDF(unique_cdf[0])
cdf_info   = sample_cdf.cdf_info()
for v in cdf_info.zVariables:
    shape = sample_cdf.varget(v).shape
    print(f'{v:40s}  shape={shape}')
```

This prints every variable in the CDF and its array shape. Example output:
```
epoch_for_cdf    shape=(34446, 5)   ← time: 34446 records, 5 energy steps
trig_counts      shape=(34446, 5, 2, 5)  ← counts per record/step/detector/channel
spacecraft_xpos  shape=(34446,)    ← scalar per record
```

### 2c. The Epoch Parsing Fix

```python
def _parse_epoch(cdf, time_var=TIME_VAR, unix_var=UNIX_VAR):
    avail = cdf.cdf_info().zVariables
    if time_var in avail:
        raw = cdf.varget(time_var)   # shape: (34446, 5)
        if raw.ndim > 1:
            raw = raw[:, 0]          # take column 0 → shape: (34446,)
        dt64 = np.asarray(cdflib.cdfepoch.to_datetime(raw))
        return pd.DatetimeIndex(dt64).tz_localize(None)
```

**Three bugs fixed here:**

| Bug | Problem | Fix |
|-----|---------|-----|
| `to_np=True` | Removed in cdflib ≥ 1.3 | Wrap result in `np.asarray()` instead |
| 2-D epoch array | `epoch_for_cdf` is `(N, 5)` — one time per energy step | Slice `[:, 0]` to get first step |
| `obs_time` fallback | Some files may not have `epoch_for_cdf` | Try Unix seconds (`obs_time`) as backup |

### 2d. The Memory-Efficient Extraction Loop

```python
def extract_daily_cdf(filepath, scalar_vars):
    ...
    arr = np.asarray(arr, dtype=np.float32)   # ← use float32, not float64
    if arr.ndim > 1:
        arr = np.nansum(arr, axis=tuple(range(1, arr.ndim)))  # ← collapse dims
    ...
    df = df.resample('1h').mean()  # ← KEY: reduce 34,000 rows → 24 rows HERE
    return df
```

**Why `float32` instead of `float64`?**
- `float64` = 8 bytes per number
- `float32` = 4 bytes per number  
- For sensor data, 7-digit precision is plenty
- **Halves memory usage** during extraction

**Why `nansum` on multi-dimensional arrays?**
- `trig_counts` shape is `(34446, 5, 2, 5)` — that's records × energy steps × detector modes × channels
- We want one number per timestamp, not a 4D tensor
- `nansum` collapses all axes after axis 0 → `(34446,)` scalar per record

**The key memory fix — resample INSIDE the loop:**

```
BEFORE (crashed with 14 GB error):
  Load file 1: 34,000 rows  →  append to list
  Load file 2: 34,000 rows  →  append to list
  ...
  Load file 694: 34,000 rows → append to list
  pd.concat(all 694 frames) = 23.5 MILLION rows → CRASH

AFTER (fixed):
  Load file 1: 34,000 rows → resample to 24 rows → append to list
  Load file 2: 34,000 rows → resample to 24 rows → append to list  
  ...
  Load file 694: 34,000 rows → resample to 24 rows → append to list
  pd.concat(694 × 24 rows) = 16,656 rows → fine ✓
```

**Memory comparison:**
- Before: `(109,000,000 rows × 18 cols × 8 bytes)` = **~14.8 GB**
- After:  `(16,656 rows × 18 cols × 4 bytes)` = **~1.2 MB**

---

## Section 3 — Time-Align Everything

### 3a. Building the Unified Time Grid

```python
time_index = pd.date_range(start=start_time, end=end_time, freq='1h')
```

This creates a **regular hourly sequence** covering all data:
```
2021-01-01 00:00:00
2021-01-01 01:00:00
2021-01-01 02:00:00
...
2025-xx-xx xx:00:00
```

### 3b. The CME Label Grid — Most Important Step

```python
def build_label_grid(time_index, cme_df, window_hours=12):
    for i, t in enumerate(ti_np):
        mask = np.abs(cme_times - t) <= delta_ns   # ← within ±12 hours?
        if mask.any():
            labels[i]  = int(sub['label'].max())
            is_halo[i] = int(sub['is_halo'].max())
```

**Why ±12 hours?**

CME events in the catalogue are logged at the **observation time** (when the CME was seen leaving the Sun). But the solar wind disturbance arrives at Earth/L1 typically **1–3 days later**.

However, since we're training a model to detect CME signatures **in the solar wind at L1**, we use the catalogue as approximate ground truth — a ±12-hour window ensures nearby CME events "label" surrounding hours, accounting for timing uncertainty.

**What the label means for ML:**
- `cme_label = 0` → normal solar wind (no CME nearby)
- `cme_label = 1` → a CME event occurred within ±12 hours of this timestamp

### 3c. Merging All Sources

```python
merged = label_grid.copy()
merged = merged.join(sw_hourly,   how='left')  # adds plasma parameters
merged = merged.join(swis_hourly, how='left')  # adds detector/position data
```

**`left` join** means: keep all hours in our time grid, add columns from right side where times match. Hours with no SWIS or solar wind data get `NaN` (filled later).

---

## Section 4 — Feature Engineering

This section creates **new meaningful features** from the raw measurements.

### 4.1 Magnetic Field Magnitude

```python
df['B_magnitude'] = np.sqrt(df['bx_gse']**2 + df['by_gse']**2 + df['bz_gse']**2)
```

The three GSE components (`Bx, By, Bz`) describe the magnetic field direction. Their combined **magnitude `|B|`** is a direction-independent measure of total field strength. CMEs often carry strong magnetic fields.

### 4.2 Alfvén Speed

```python
df['alfven_speed'] = 21.8 * df['B_magnitude'] / np.sqrt(df['proton_density'].clip(lower=0.001))
```

The **Alfvén speed** is the speed at which magnetic disturbances travel through the plasma:

```
V_A = |B| / √(μ₀ × ρ)
```

Where `21.8` is a unit-conversion constant. A high Alfvén speed relative to plasma speed indicates a magnetically-dominated region — typical of CME-driven sheath regions.

### 4.3 Dynamic Pressure

```python
df['dynamic_pressure'] = 1.67e-6 * df['proton_density'] * df['plasma_speed_kmps']**2
```

Ram pressure of the solar wind:  `P = ½ × m_p × n × v²`

CMEs compress the magnetosphere when high-pressure plasma arrives. This is directly linked to geomagnetic storm intensity.

### 4.4 Bz Southward Indicator

```python
df['bz_southward'] = df['bz_gse'].clip(upper=0).abs()
```

`bz_gse` is the **north-south component** of the magnetic field:
- When `Bz > 0` (northward) → no energy transfer → `bz_southward = 0`
- When `Bz < 0` (southward) → magnetic reconnection with Earth's field → geomagnetic storm
- CMEs often carry sustained southward Bz — this is the primary storm driver

`.clip(upper=0)` zeros out positive values, `.abs()` makes negatives positive.

### 4.5 Alfvénic Mach Number

```python
df['mach_alfven'] = df['plasma_speed_kmps'] / df['alfven_speed'].clip(lower=0.001)
```

The ratio of plasma speed to Alfvén speed (`M_A`):
- `M_A > 1` → super-Alfvénic flow (can drive shocks)
- CME-driven shocks occur when `M_A >> 1` ahead of the ejecta

### 4.6 Rolling Statistics

```python
for var in ROLL_VARS:
    for w in [3, 6, 12]:
        df[f'{var}_mean{w}h'] = df[var].rolling(w, min_periods=1).mean()
        df[f'{var}_std{w}h']  = df[var].rolling(w, min_periods=1).std()
    df[f'{var}_delta1h'] = df[var].diff(1)
```

For each key variable, we compute:

| Feature | Meaning |
|---------|---------|
| `_mean3h` | Average over past 3 hours → smooths noise |
| `_mean6h` | Average over past 6 hours → medium-term trend |
| `_mean12h` | Average over past 12 hours → long-term baseline |
| `_std3h` | Variability over 3 hours → detects sudden fluctuations |
| `_delta1h` | Change from last hour → rate-of-change signal |

This is crucial for CME detection — the **trend and rate of change** is often more diagnostic than the absolute value.

---

## Section 5 — Quality Checks

### Missing Value Analysis

```python
missing = df.isnull().mean().sort_values(ascending=False) * 100
```

Calculates what **percentage of each column is NaN**:
- SWIS data only covers 2024–2025, so SWIS columns will be ~70% missing for the 2021-2024 solar wind period
- Solar wind CSV only covers 2021-2024, so those columns will be ~NaN for 2025

### Dropping High-Missing Columns

```python
drop_cols = [c for c in missing[missing > HIGH_MISS_THRESH].index
             if c not in ['cme_label', 'is_halo']]
df.drop(columns=drop_cols, ...)
```

If a column is missing **>60% of values**, it's not useful for training — dropped entirely. Label columns are always protected.

### Imputation Strategy

```python
df[feat_cols] = df[feat_cols].ffill().bfill()          # temporal gap filling
df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())  # remaining NaNs
```

1. **`ffill()` (forward fill)** → propagates last known value forward (handles short data gaps in time series)
2. **`bfill()` (backward fill)** → propagates next known value backward (handles leading NaNs)
3. **`median fill`** → any remaining NaNs filled with the column's median (robust to outliers)

---

## Section 6 — Train / Validation / Test Split

### Why Chronological, Not Random?

```python
n         = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train_df = df.iloc[:train_end]   # earliest 70%
val_df   = df.iloc[train_end:val_end]  # middle 15%
test_df  = df.iloc[val_end:]     # latest 15%
```

With time-series data, **random splitting leaks future information into training**. If a random split picks a window from 2025 for training and a 2023 window for testing, the model is effectively tested on the past — not realistic.

Chronological splitting ensures the model is always evaluated on data **it has never seen in the future**.

### Feature Scaling

```python
scaler  = StandardScaler()
X_train = scaler.fit_transform(train_df[FEATURE_COLS])  # learn mean & std from train only
X_val   = scaler.transform(val_df[FEATURE_COLS])        # apply same scaling
X_test  = scaler.transform(test_df[FEATURE_COLS])       # apply same scaling
```

**StandardScaler** transforms each feature to zero mean and unit variance:
```
X_scaled = (X - mean) / std
```

**Critical rule:** `fit_transform` on train, `transform` only on val/test. If you fit on val or test data, you've "seen the future" (data leakage).

---

## Section 7 — Export

| File | Contents | Used for |
|------|----------|----------|
| `full_merged_dataset.csv` | All data with labels | Full analysis |
| `train.csv / val.csv / test.csv` | Un-scaled splits | EDA, SHAP explainability |
| `X_train.npy / y_train.npy` | Scaled numpy arrays | Direct model training |
| `X_val.npy / y_val.npy` | Scaled numpy arrays | Hyperparameter tuning |
| `X_test.npy / y_test.npy` | Scaled numpy arrays | Final evaluation |
| `scaler.pkl` | Fitted StandardScaler | Inference (scale new data) |
| `feature_list.csv` | List of feature names | Mapping array columns back to names |

---

## Section 8 — LSTM Sequence Dataset (Optional)

```python
def make_sequences(X, y, window=24, horizon=6):
    for i in range(len(X) - window - horizon + 1):
        Xs.append(X[i : i + window])          # 24 hours of past data
        ys.append(y[i + window + horizon - 1]) # label 6 hours ahead
```

This creates **sliding windows** for sequence models (LSTM, Transformer):

```
Window 1: hours [0..23]   → predict label at hour 29
Window 2: hours [1..24]   → predict label at hour 30
...
```

- `window=24` → model sees 24 hours of history
- `horizon=6` → predicts whether a CME will arrive in 6 hours
- This turns the problem into a **forecasting** task (not just detection)

---

## 🔄 Complete Data Flow Diagram

```
solar_wind_2021_2024.csv          cme_labels_2021_2024.csv
        │                          cme_labels_2024_2025.csv
        │                                    │
        ▼                                    ▼
   sw_hourly (hourly)              cme_all (event catalogue)
        │                                    │
        │                                    ▼
SWIS .cdf files              build_label_grid() → hourly 0/1 labels
        │                                    │
        ▼                                    │
extract_daily_cdf()                          │
  → resample to hourly                       │
  → sum multi-dim arrays                     │
        │                                    │
        ▼                                    │
   swis_hourly (~16K rows)                   │
        │                                    │
        └────────────┬───────────────────────┘
                     ▼
              merged DataFrame
                     │
                     ▼
          feature engineering
          (|B|, Alfvén, pressure, rolling stats)
                     │
                     ▼
          quality checks + imputation
                     │
                     ▼
          chronological 70/15/15 split
                     │
               StandardScaler
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       train.csv  val.csv   test.csv
       X/y_train  X/y_val   X/y_test
                  scaler.pkl
```
