"""
physics_model.py  --  Digital twin, STEP 2 (physics model) + STEP 3 (residual analysis)

STEP 2 (unchanged physics)
    Predicts what a HEALTHY engine should read, using only the engine INPUTS
    (RPM, load, ambient temperature/pressure).  Knobs are calibrated on rows
    labelled fault_type == "normal".

STEP 3 (new)
    Compares MEASURED vs PREDICTED, and answers:
      "How far is the engine from the healthy twin, and how abnormal /
       persistent / drifting is that deviation?"
    It does NOT diagnose faults (that is Step 4).

    residual = measured - predicted
    z        = (residual - healthy_mean_residual) / healthy_std_residual
    + rolling statistics, trend, persistence  ->  anomaly score  ->  status

HOW TO RUN
    python physics_model.py engine_dataset.csv

OUTPUTS
    engine_with_predictions.csv   measured + twin predictions (as before)
    twin_check.png                measured vs predicted (as before)
    engine_residuals.csv          STEP 3, long format (1 row per timestamp x sensor)
    engine_residual_features.csv  STEP 3, wide format (1 row per timestamp) -> Step 4 input
    residual_analysis.png         STEP 3 diagnostic plots

ALL STEP-3 THRESHOLDS ARE PROTOTYPE ASSUMPTIONS, NOT AVIATION STANDARDS.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # headless: just write PNG files
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.optimize import least_squares

RPM_RATED = 5500.0   # rated engine speed. Change this to your engine's value.

# Measured columns in the CSV that the twin predicts (prediction = name + "_pred")
CHANNELS = [
    "cht_c", "egt_c", "oil_temperature_c", "oil_pressure_bar",
    "fuel_flow_l_h", "fuel_pressure_bar", "vibration_g", "voltage_v",
]

# The model's "knobs".  name: (starting guess, lowest allowed, highest allowed)
# The starting guesses are rough; calibrate() tunes them to your data.
PARAMS = {
    # Exhaust gas temperature: target = ambient + base + a*rpm + b*power
    "egt_base":  (220.0, 0.0, 600.0),
    "egt_rpm":   (250.0, 0.0, 1500.0),
    "egt_power": (350.0, 0.0, 1500.0),
    "tau_egt":   (6.0, 1.0, 60.0),            # seconds to react (fast)
    # Cylinder head temperature
    "cht_base":  (40.0, 0.0, 200.0),
    "cht_rpm":   (35.0, 0.0, 300.0),
    "cht_power": (90.0, 0.0, 500.0),
    "tau_cht":   (90.0, 10.0, 600.0),         # slow (metal mass)
    # Oil temperature
    "oilt_base":  (35.0, 0.0, 150.0),
    "oilt_rpm":   (25.0, 0.0, 300.0),
    "oilt_power": (40.0, 0.0, 300.0),
    "tau_oilt":   (150.0, 20.0, 1200.0),      # slowest
    # Oil pressure (bar): pump pressure levels off with RPM, drops when oil is hot
    "oilp_max": (2.9, 0.5, 8.0),
    "oilp_rpm": (400.0, 50.0, 3000.0),
    "oilp_hot": (0.15, 0.0, 0.6),
    # Fuel flow (L/h)
    "ff_idle":      (0.2, 0.0, 5.0),
    "ff_rpm":       (9.5, 0.0, 80.0),
    "ff_rpm_scale": (400.0, 50.0, 3000.0),
    "ff_power":     (30.0, 0.0, 300.0),
    # Fuel pressure (bar)
    "fp_max":   (0.3, 0.0, 6.0),
    "fp_rpm":   (100.0, 10.0, 2000.0),
    "fp_power": (0.5, 0.0, 3.0),
    # Vibration (g): imbalance force grows with RPM squared
    "vib_base": (0.01, 0.0, 0.5),
    "vib_rpm2": (3.0, 0.0, 30.0),
    # Battery / alternator voltage
    "volt_batt": (12.0, 9.0, 13.0),
    "volt_reg":  (13.9, 12.5, 15.0),
    "volt_rpm":  (250.0, 30.0, 3000.0),
}


def _lag(target, tau, dt, y0):
    """First-order lag: the value moves toward 'target' with time constant tau."""
    a = 1.0 - np.exp(-dt / tau)
    return lfilter([a], [1.0, -(1.0 - a)], target, zi=[(1.0 - a) * y0])[0]


def _simulate(p, x, init):
    """Core physics. p = knob values, x = input arrays, init = starting temps."""
    rpm, load, t_amb, p_amb, dt = x["rpm"], x["load"], x["t_amb"], x["p_amb"], x["dt"]

    rpm_n = rpm / RPM_RATED                              # 0..1 speed
    power = rpm_n * load / 100.0                         # rough power fraction
    sigma = (p_amb / 1013.25) * (288.15 / (t_amb + 273.15))   # air density ratio
    cooling = np.clip(sigma, 0.2, 1.5) ** 0.6            # thin air cools worse
    running = 1.0 - np.exp(-rpm / 100.0)                 # 0 when engine is stopped

    egt_target = t_amb + running * (p["egt_base"] + p["egt_rpm"] * rpm_n + p["egt_power"] * power)
    cht_target = t_amb + running * (p["cht_base"] + p["cht_rpm"] * rpm_n + p["cht_power"] * power) / cooling
    oilt_target = t_amb + running * (p["oilt_base"] + p["oilt_rpm"] * rpm_n + p["oilt_power"] * power)

    out = {
        "egt_c": _lag(egt_target, p["tau_egt"], dt, init["egt"]),
        "cht_c": _lag(cht_target, p["tau_cht"], dt, init["cht"]),
        "oil_temperature_c": _lag(oilt_target, p["tau_oilt"], dt, init["oilt"]),
    }
    hot = np.clip(1.0 - p["oilp_hot"] * (out["oil_temperature_c"] - 20.0) / 80.0, 0.3, 1.1)
    out["oil_pressure_bar"] = p["oilp_max"] * (1 - np.exp(-rpm / p["oilp_rpm"])) * hot
    out["fuel_flow_l_h"] = (p["ff_idle"] + p["ff_rpm"] * (1 - np.exp(-rpm / p["ff_rpm_scale"]))
                            + p["ff_power"] * power)
    out["fuel_pressure_bar"] = p["fp_max"] * (1 - np.exp(-rpm / p["fp_rpm"])) + p["fp_power"] * power
    out["vibration_g"] = p["vib_base"] + p["vib_rpm2"] * rpm_n ** 2
    out["voltage_v"] = p["volt_batt"] + (p["volt_reg"] - p["volt_batt"]) * (1 - np.exp(-rpm / p["volt_rpm"]))
    return out


class EngineTwin:
    def __init__(self, params=None):
        self.p = {k: v[0] for k, v in PARAMS.items()}
        if params:
            self.p.update(params)
        self.scale = None

    # ---------- reading the CSV ----------
    @staticmethod
    def _inputs(df):
        t = pd.to_datetime(df["timestamp"], utc=True)
        dt = t.diff().dt.total_seconds().median()
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0
        return {
            "rpm": np.clip(df["rpm"].to_numpy(float), 0, None),
            "load": df["engine_load_percent"].to_numpy(float),
            "t_amb": df["ambient_temperature_c"].to_numpy(float),
            "p_amb": df["ambient_pressure_hpa"].to_numpy(float),
            "dt": float(dt),
        }

    @staticmethod
    def _init_state(df, x, use_first_row):
        """Start the twin from the engine's first reading (this is the 'sync')."""
        init = {}
        for key, col in (("egt", "egt_c"), ("cht", "cht_c"), ("oilt", "oil_temperature_c")):
            v = df[col].iloc[0] if (use_first_row and col in df) else np.nan
            init[key] = float(v) if np.isfinite(v) else float(x["t_amb"][0])
        return init

    @staticmethod
    def _healthy_mask(df):
        if "fault_type" in df:
            return (df["fault_type"] == "normal").to_numpy()
        return np.ones(len(df), dtype=bool)

    # ---------- predict ----------
    def predict(self, df, use_first_row=True):
        x = self._inputs(df)
        sim = _simulate(self.p, x, self._init_state(df, x, use_first_row))
        pred = pd.DataFrame({c + "_pred": sim[c] for c in CHANNELS}, index=df.index)
        # altitude is not in the CSV, so estimate it from ambient pressure
        pred["altitude_est_m"] = 44330.0 * (1.0 - (x["p_amb"] / 1013.25) ** 0.1903)
        return pred

    # ---------- tune the knobs on healthy data ----------
    def calibrate(self, df, use_first_row=True, verbose=True):
        x = self._inputs(df)
        init = self._init_state(df, x, use_first_row)
        ok = self._healthy_mask(df)
        meas = {c: df[c].to_numpy(float) for c in CHANNELS}
        use = {c: ok & np.isfinite(meas[c]) for c in CHANNELS}   # healthy rows only
        self.scale = {c: max(float(np.nanstd(meas[c][use[c]])), 1e-3) for c in CHANNELS}

        names = list(PARAMS)
        lo = np.array([PARAMS[n][1] for n in names])
        hi = np.array([PARAMS[n][2] for n in names])
        x0 = np.clip([self.p[n] for n in names], lo, hi)

        def residuals(theta):
            sim = _simulate(dict(zip(names, theta)), x, init)
            parts = []
            for c in CHANNELS:
                r = np.nan_to_num((sim[c] - meas[c]) / self.scale[c], nan=0.0, posinf=1e3, neginf=-1e3)
                parts.append(r[use[c]])
            return np.concatenate(parts)

        start_cost = 0.5 * np.sum(residuals(x0) ** 2)
        fit = least_squares(residuals, x0, bounds=(lo, hi), loss="soft_l1",
                            x_scale="jac", max_nfev=300)
        self.p = dict(zip(names, fit.x))
        if verbose:
            print(f"Calibration used {int(ok.sum())} healthy rows out of {len(df)}.")
            print(f"Fit error score: {start_cost:.1f} -> {0.5 * np.sum(fit.fun ** 2):.1f} (lower is better)")
        return self

    # ---------- how good is it? ----------
    # NOTE: your pasted file was cut off inside this method; this tail is
    # re-created.  If your original report() differs, keep yours.
    def report(self, df, pred):
        ok = self._healthy_mask(df)
        rows = []
        for c in CHANNELS:
            err = (df[c].to_numpy(float) - pred[c + "_pred"].to_numpy(float))[ok]
            err = err[np.isfinite(err)]
            if err.size == 0:
                rows.append({"channel": c, "bias": np.nan, "mae": np.nan, "rmse": np.nan})
                continue
            rows.append({"channel": c, "bias": err.mean(), "mae": np.abs(err).mean(),
                         "rmse": float(np.sqrt(np.mean(err ** 2)))})
        return pd.DataFrame(rows).set_index("channel")

    def fault_report(self, df, pred):
        """Mean raw residual per fault_type and channel (quick sanity view)."""
        if "fault_type" not in df:
            return pd.DataFrame()
        res = pd.DataFrame({c: df[c].to_numpy(float) - pred[c + "_pred"].to_numpy(float)
                            for c in CHANNELS}, index=df.index)
        return res.groupby(df["fault_type"]).mean()


# =============================================================================
#                          STEP 3 : RESIDUAL ANALYSIS
# =============================================================================
# ---- PROTOTYPE ASSUMPTIONS (tunable; NOT aviation standards) -----------------
ROLL_WINDOW_S        = 60.0    # rolling window for residual mean/std/|z| (seconds)
TREND_WINDOW_S       = 180.0   # window for the drift (slope) estimate (seconds)
MIN_ROLL_SAMPLES     = 5       # fewer samples in window -> rolling stats = "not enough history"
MIN_TREND_SAMPLES    = 8

WARMUP_S             = 120.0   # first N seconds of the file are treated as start-up
TRANSIENT_WINDOW_S   = 30.0    # look-back window used to detect RPM changes
RPM_STEP_FRAC        = 0.08    # RPM swing > 8% of rated inside the window = transient
TRANSIENT_CONDITION_KEYWORDS = ("startup", "start_up", "warm", "transient")  # matched in operating_condition
TRANSIENT_WEIGHT     = 0.6     # score multiplier during transients (damped, not ignored)

MIN_BASELINE_SAMPLES = 30      # min healthy steady rows needed for a baseline
STD_REL_FLOOR        = 0.002   # std floor = 0.2% of the healthy p5..p95 range of the sensor
STD_ABS_FLOOR        = 1e-6    # absolute floor so we never divide by ~0
MIN_EFFECT_FRAC      = 0.01    # a deviation smaller than 1% of range (or 1 sigma) is "practically insignificant"

Z_PERSIST            = 2.5     # |z| above this counts as "abnormal sample" for persistence
Z_SAT                = 8.0     # |z| that saturates the instantaneous component
ROLL_Z_SAT           = 5.0     # rolling mean |z| that saturates the rolling component
PERSIST_SAT_S        = 60.0    # seconds of continuous abnormality that saturates persistence
TREND_SAT_Z_PER_MIN  = 2.0     # drift rate (z per minute) that saturates the trend component
TREND_MIN_ROLL_Z     = 1.0     # trend only counts if rolling mean |z| is at least this
W_INST, W_ROLL, W_PERS, W_TREND = 0.20, 0.35, 0.30, 0.15   # component weights (sum = 1)

WATCH_SCORE, ANOMALOUS_SCORE, SEVERE_SCORE = 25.0, 50.0, 75.0   # score thresholds (0..100)
MIN_PERSIST_ANOMALOUS_S = 10.0   # ANOMALOUS needs this many seconds of continuous abnormality
MIN_PERSIST_SEVERE_S    = 30.0
STATUS_NAMES = ["NORMAL", "WATCH", "ANOMALOUS", "SEVERE"]


def load_dataset(path):
    """Read the CSV, sort by time, make sure numeric columns are numeric."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    for c in CHANNELS + ["rpm", "engine_load_percent", "ambient_temperature_c",
                         "ambient_pressure_hpa"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def detect_transients(df, ts, t_sec):
    """True where the twin/engine mismatch is expected to be larger than usual:
    (a) warm-up period at the start of the file,
    (b) operating_condition text that looks like start-up/warm-up/transient,
    (c) RPM has swung by more than RPM_STEP_FRAC*RPM_RATED within the last TRANSIENT_WINDOW_S."""
    n = len(df)
    warm = t_sec < WARMUP_S
    cond = np.zeros(n, dtype=bool)
    if "operating_condition" in df:
        oc = df["operating_condition"].astype(str).str.lower()
        for kw in TRANSIENT_CONDITION_KEYWORDS:
            cond |= oc.str.contains(kw, regex=False).to_numpy()
    rpm = pd.Series(df["rpm"].to_numpy(float), index=ts)
    w = f"{int(round(TRANSIENT_WINDOW_S))}s"
    swing = rpm.rolling(w, min_periods=2).max() - rpm.rolling(w, min_periods=2).min()
    step = swing.fillna(0.0).to_numpy() > RPM_STEP_FRAC * RPM_RATED
    return warm | cond | step


def healthy_baseline(df, pred, healthy, transient, verbose=True):
    """Per-sensor healthy residual statistics.  ONLY rows with fault_type=='normal'
    are used; steady-state rows are preferred (start-up/transients excluded)."""
    base = {}
    for c in CHANNELS:
        meas = df[c].to_numpy(float)
        res = meas - pred[c + "_pred"].to_numpy(float)
        fin = np.isfinite(res)
        m = healthy & fin & ~transient
        basis = "healthy_steady"
        if m.sum() < MIN_BASELINE_SAMPLES:          # fallback: allow transient healthy rows
            m, basis = healthy & fin, "healthy_all"
        mv = meas[healthy & np.isfinite(meas)]
        span = float(np.percentile(mv, 95) - np.percentile(mv, 5)) if mv.size > 1 else 1.0
        span = max(span, 1e-6)
        std_floor = max(STD_ABS_FLOOR, STD_REL_FLOOR * span)
        if m.sum() >= 2:
            mu = float(res[m].mean())
            sd = float(res[m].std(ddof=1))
        else:                                        # no healthy data at all
            mu, sd, basis = 0.0, std_floor, "none(default)"
        sd_used = max(sd, std_floor)
        base[c] = {"mean": mu, "std": sd_used, "std_raw": sd, "n": int(m.sum()), "basis": basis,
                   "min_effect": max(MIN_EFFECT_FRAC * span, sd_used)}
    if verbose:
        print("\nHealthy residual baseline (measured - predicted):")
        print(pd.DataFrame(base).T[["mean", "std", "std_raw", "n", "basis"]].to_string())
    return base


def _run_persistence_seconds(flag, t_sec, dt):
    """Seconds the current uninterrupted run of flagged samples has lasted (0 if not flagged)."""
    s = pd.Series(t_sec)
    gid = (~flag).cumsum()
    first = s.where(flag).groupby(gid).transform("min")
    return np.where(flag, (s - first).to_numpy() + dt, 0.0)


def _sensor_frame(c, df, pred, ts, t_sec, dt, base, transient):
    meas = df[c].to_numpy(float)
    predv = pred[c + "_pred"].to_numpy(float)
    resid = meas - predv
    mu, sd, min_eff = base["mean"], base["std"], base["min_effect"]
    dev = resid - mu
    z = dev / sd                                      # NaN stays NaN (missing data)
    valid = np.isfinite(z)

    rs, zs = pd.Series(resid, index=ts), pd.Series(z, index=ts)
    rw, tw = f"{int(round(ROLL_WINDOW_S))}s", f"{int(round(TREND_WINDOW_S))}s"
    roll_mean = rs.rolling(rw, min_periods=MIN_ROLL_SAMPLES).mean()
    roll_std = rs.rolling(rw, min_periods=MIN_ROLL_SAMPLES).std()
    roll_abs_z = zs.abs().rolling(rw, min_periods=MIN_ROLL_SAMPLES).mean()
    roll_mean_z = zs.rolling(rw, min_periods=MIN_ROLL_SAMPLES).mean()

    # --- trend: least-squares slope of z vs time over TREND_WINDOW_S, from rolling moments
    tt = pd.Series(np.where(valid, t_sec, np.nan), index=ts)
    mt = tt.rolling(tw, min_periods=MIN_TREND_SAMPLES).mean()
    mz = zs.rolling(tw, min_periods=MIN_TREND_SAMPLES).mean()
    mtz = (tt * zs).rolling(tw, min_periods=MIN_TREND_SAMPLES).mean()
    mt2 = (tt * tt).rolling(tw, min_periods=MIN_TREND_SAMPLES).mean()
    var_t, cov_tz = mt2 - mt ** 2, mtz - mt * mz
    slope = (cov_tz / var_t.where(var_t > 1e-6)) * 60.0          # z per minute (NaN if unknown)
    slope_np = slope.to_numpy()

    # --- persistence
    persist_s = _run_persistence_seconds(np.abs(np.nan_to_num(z, nan=0.0)) > Z_PERSIST, t_sec, dt)

    # --- score components, each 0..1 (NaN -> 0 = "no evidence")
    def clip01(a):
        return np.clip(np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    c_inst = clip01(np.abs(z) / Z_SAT)
    c_roll = clip01(roll_abs_z.to_numpy() / ROLL_Z_SAT)
    c_pers = clip01(persist_s / PERSIST_SAT_S)
    drift_away = slope_np * np.sign(np.nan_to_num(roll_mean_z.to_numpy()))   # >0: moving away from healthy
    trend_ok = np.nan_to_num(roll_abs_z.to_numpy()) >= TREND_MIN_ROLL_Z
    c_trend = clip01(drift_away / TREND_SAT_Z_PER_MIN) * trend_ok

    # practical-significance gate: tiny deviations (vs sensor range) can't score high
    roll_dev = roll_mean.to_numpy() - mu
    mag = np.where(np.isfinite(roll_dev), np.abs(roll_dev), np.nan_to_num(np.abs(dev)))
    gate = clip01(mag / min_eff)

    score = 100.0 * (W_INST * c_inst + W_ROLL * c_roll + W_PERS * c_pers + W_TREND * c_trend) * gate
    score = score * np.where(transient, TRANSIENT_WEIGHT, 1.0)

    level = np.select(
        [(score >= SEVERE_SCORE) & (persist_s >= MIN_PERSIST_SEVERE_S),
         (score >= ANOMALOUS_SCORE) & (persist_s >= MIN_PERSIST_ANOMALOUS_S),
         score >= WATCH_SCORE], [3, 2, 1], default=0).astype(int)

    return pd.DataFrame({
        "row_id": np.arange(len(df)),
        "timestamp": ts.to_numpy(),
        "sensor": c,
        "measured": meas,
        "predicted": predv,
        "residual": resid,
        "healthy_resid_mean": mu,
        "healthy_resid_std": sd,
        "z": z,
        "roll_mean_resid": roll_mean.to_numpy(),
        "roll_std_resid": roll_std.to_numpy(),
        "roll_std_ratio": (roll_std / sd).to_numpy(),         # ~1 healthy; >>1 = noisier than healthy
        "roll_mean_abs_z": roll_abs_z.to_numpy(),
        "trend_z_per_min": slope_np,
        "persist_s": persist_s,
        "score_instant": c_inst, "score_rolling": c_roll,
        "score_persist": c_pers, "score_trend": c_trend,
        "anomaly_score": score,
        "status_level": level,
        "anomaly_status": np.array(STATUS_NAMES)[level],
        "data_valid": valid,
        "history_ok": np.isfinite(roll_abs_z.to_numpy()),
        "in_transient": transient,
    })


def residual_analysis(df, pred, verbose=True):
    """Full Step-3 pipeline.  Returns (long_df, wide_df)."""
    ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
    t_sec = (ts - ts[0]).total_seconds().to_numpy(float)
    dt = pd.Series(t_sec).diff().median()
    dt = float(dt) if np.isfinite(dt) and dt > 0 else 1.0
    healthy = EngineTwin._healthy_mask(df)
    transient = detect_transients(df, ts, t_sec)
    if verbose:
        print(f"\nSampling interval (median): {dt:g} s | transient rows: {int(transient.sum())}/{len(df)}")
    base = healthy_baseline(df, pred, healthy, transient, verbose)

    frames = {c: _sensor_frame(c, df, pred, ts, t_sec, dt, base[c], transient) for c in CHANNELS}
    long = pd.concat(frames.values(), ignore_index=True)

    # ground truth (verification only -- NEVER use as ML input features)
    for src, dst in (("operating_condition", "gt_operating_condition"),
                     ("fault_type", "gt_fault_type"), ("fault_severity", "gt_fault_severity")):
        if src in df:
            long[dst] = np.tile(df[src].to_numpy(), len(CHANNELS))

    # wide format: one row per timestamp, per-sensor feature columns
    feat = ["residual", "z", "roll_mean_resid", "roll_std_ratio", "roll_mean_abs_z",
            "trend_z_per_min", "persist_s", "anomaly_score", "status_level"]
    wide = pd.DataFrame({"timestamp": ts.to_numpy(), "in_transient": transient})
    for c in CHANNELS:
        for f in feat:
            wide[f"{c}__{f}"] = frames[c][f].to_numpy()
    scores = np.column_stack([frames[c]["anomaly_score"].to_numpy() for c in CHANNELS])
    levels = np.column_stack([frames[c]["status_level"].to_numpy() for c in CHANNELS])
    wide["engine_max_score"] = scores.max(axis=1)
    wide["engine_max_status_level"] = levels.max(axis=1)
    wide["n_sensors_watch_or_worse"] = (levels >= 1).sum(axis=1)
    wide["n_sensors_anomalous_or_worse"] = (levels >= 2).sum(axis=1)
    wide["top_sensor"] = np.where(scores.max(axis=1) > 0, np.array(CHANNELS)[scores.argmax(axis=1)], "none")
    for src, dst in (("operating_condition", "gt_operating_condition"),
                     ("fault_type", "gt_fault_type"), ("fault_severity", "gt_fault_severity")):
        if src in df:
            wide[dst] = df[src].to_numpy()
    return long, wide


def print_verification(long, wide):
    """Text summary: healthy data should stay quiet, faulty data should light up."""
    if "gt_fault_type" not in long:
        print("No fault_type column: skipping verification summary.")
        return
    h = long["gt_fault_type"] == "normal"
    rows = []
    for c in CHANNELS:
        s = long[long["sensor"] == c]
        hh = s["gt_fault_type"] == "normal"
        rows.append({
            "sensor": c,
            "healthy_mean|z|": s.loc[hh, "z"].abs().mean(),
            "faulty_mean|z|": s.loc[~hh, "z"].abs().mean(),
            "healthy_%>=ANOM (false alarms)": 100 * (s.loc[hh, "status_level"] >= 2).mean(),
            "faulty_%>=ANOM": 100 * (s.loc[~hh, "status_level"] >= 2).mean(),
        })
    print("\nVerification per sensor (healthy should be low, faulty higher):")
    print(pd.DataFrame(rows).set_index("sensor").round(2).to_string())

    g = wide.groupby("gt_fault_type")
    summ = pd.DataFrame({
        "rows": g.size(),
        "%rows_any_sensor>=WATCH": 100 * g["engine_max_status_level"].apply(lambda s: (s >= 1).mean()),
        "%rows_any_sensor>=ANOM": 100 * g["engine_max_status_level"].apply(lambda s: (s >= 2).mean()),
        "most_common_top_sensor": g["top_sensor"].agg(
            lambda s: s[s != "none"].mode().iat[0] if (s != "none").any() else "none"),
    })
    print("\nVerification per fault_type (engine-level):")
    print(summ.round(1).to_string())


def plot_residual_analysis(long, path="residual_analysis.png"):
    """8 sensors x 3 columns: measured vs twin | residual (+healthy band) | anomaly score."""
    n = len(CHANNELS)
    fig, axes = plt.subplots(n, 3, figsize=(20, 2.8 * n), sharex=True)
    has_gt = "gt_fault_type" in long

    def shade(ax, x, mask, color, label):
        ax.fill_between(x, 0, 1, where=mask, transform=ax.get_xaxis_transform(),
                        color=color, alpha=0.18, lw=0, label=label)

    for i, c in enumerate(CHANNELS):
        s = long[long["sensor"] == c].reset_index(drop=True)
        x = (s["timestamp"] - s["timestamp"].iloc[0]).dt.total_seconds().to_numpy() / 60.0
        fault = (s["gt_fault_type"] != "normal").to_numpy() if has_gt else np.zeros(len(s), bool)

        a = axes[i, 0]
        a.plot(x, s["measured"], lw=0.6, label="measured")
        a.plot(x, s["predicted"], lw=0.8, label="twin")
        if has_gt:
            shade(a, x, fault, "red", "fault (truth)")
        a.set_ylabel(c, fontsize=8)
        if i == 0:
            a.set_title("Measured vs Digital Twin"); a.legend(fontsize=7, loc="upper right")

        a = axes[i, 1]
        mu, sd = s["healthy_resid_mean"].iat[0], s["healthy_resid_std"].iat[0]
        a.plot(x, s["residual"], lw=0.5, color="tab:purple", label="residual")
        a.plot(x, s["roll_mean_resid"], lw=1.0, color="k", label="rolling mean")
        a.axhspan(mu - 3 * sd, mu + 3 * sd, color="green", alpha=0.12, label="healthy ±3σ")
        if has_gt:
            shade(a, x, fault, "red", None)
        if i == 0:
            a.set_title("Residual (measured - twin)"); a.legend(fontsize=7, loc="upper right")

        a = axes[i, 2]
        a.plot(x, s["anomaly_score"], lw=0.8, color="tab:orange")
        for thr, col in ((WATCH_SCORE, "gold"), (ANOMALOUS_SCORE, "darkorange"), (SEVERE_SCORE, "red")):
            a.axhline(thr, color=col, lw=0.7, ls="--")
        shade(a, x, (s["status_level"] >= 2).to_numpy(), "orange", "status >= ANOMALOUS")
        if has_gt:
            shade(a, x, fault, "red", None)
        a.set_ylim(0, 100)
        if i == 0:
            a.set_title("Anomaly score (0-100)"); a.legend(fontsize=7, loc="upper right")
    for a in axes[-1]:
        a.set_xlabel("time (minutes)")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def plot_twin_check(df, pred, path="twin_check.png"):
    """Simple measured-vs-predicted picture (Step 2 sanity check)."""
    t = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().to_numpy() / 60.0
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(12, 2.2 * len(CHANNELS)), sharex=True)
    for a, c in zip(axes, CHANNELS):
        a.plot(t, df[c], lw=0.5, label="measured")
        a.plot(t, pred[c + "_pred"], lw=0.8, label="twin")
        a.set_ylabel(c, fontsize=8)
    axes[0].legend(fontsize=7)
    axes[-1].set_xlabel("time (minutes)")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "engine_dataset.csv"
    df = load_dataset(path)

    twin = EngineTwin().calibrate(df)
    pred = twin.predict(df)

    print("\nTwin accuracy on healthy rows:")
    print(twin.report(df, pred).round(4).to_string())
    fr = twin.fault_report(df, pred)
    if not fr.empty:
        print("\nMean raw residual per fault_type:")
        print(fr.round(3).to_string())

    pd.concat([df, pred], axis=1).to_csv("engine_with_predictions.csv", index=False)
    plot_twin_check(df, pred)

    # ---- STEP 3 ----
    long, wide = residual_analysis(df, pred)
    long.to_csv("engine_residuals.csv", index=False)
    wide.to_csv("engine_residual_features.csv", index=False)
    plot_residual_analysis(long)
    print_verification(long, wide)
    print("\nSaved: engine_with_predictions.csv, twin_check.png, engine_residuals.csv, "
          "engine_residual_features.csv, residual_analysis.png")


if __name__ == "__main__":
    main()
