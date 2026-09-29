"""
ml_features.py  --  STAGE 4a: feature engineering for the AI/ML layer

Reads the existing, already-working Stage 3 outputs:
    engine_residual_features.csv   (wide: 1 row per timestamp)
    engine_residuals.csv           (long: 1 row per timestamp x sensor)

Does NOT touch physics_model.py, does NOT recompute residuals.
Only adds a small number of extra, well-justified features on top of what
Stage 3 already produced, then writes ml_features.csv.

FEATURES ADDED HERE (beyond what physics_model.py already provides)
---------------------------------------------------------------------
1. rate_of_change: residual(t) - residual(t-1), per sensor
   -> captures how fast a deviation is moving right now, complements the
      slower 180s trend slope already computed in Stage 3.
2. pct_error: residual / |predicted|  (with epsilon floor)
   -> normalizes residual magnitude relative to the operating point, so a
      2 bar error at low RPM and at high RPM are comparable.
3. cross-sensor rolling correlation for physically-linked sensor pairs
   (egt vs cht, oil_pressure vs oil_temperature) over the same rolling
   window Stage 3 already uses.
   -> a real thermal/lubrication fault usually moves TWO sensors together;
      a lone sensor drift looks different from a coupled physical fault.

Everything from Stage 3 (residual, z, roll_mean_resid, roll_std_ratio,
roll_mean_abs_z, trend_z_per_min, persist_s, anomaly_score, status_level) is
carried through unchanged and is also part of the final ML feature set.
"""

import numpy as np
import pandas as pd
import physics_model as pm   # reuse CHANNELS + rolling-window constants, no duplication

# physically-linked sensor pairs worth correlating (prototype choice, documented)
CORRELATION_PAIRS = [
    ("egt_c", "cht_c"),                  # combustion/thermal: both should move together
    ("oil_pressure_bar", "oil_temperature_c"),  # lubrication: pressure drops as oil heats
]

EPS = 1e-6  # floor for pct_error denominator


def _rate_of_change(long_df):
    """residual(t) - residual(t-1) per sensor, using the actual timestamp order."""
    long_df = long_df.sort_values(["sensor", "timestamp"])
    roc = long_df.groupby("sensor")["residual"].diff().fillna(0.0)
    long_df = long_df.assign(rate_of_change=roc)
    return long_df


def _pct_error(long_df):
    denom = long_df["predicted"].abs().clip(lower=EPS)
    return (long_df["residual"] / denom).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _cross_correlations(df_wide, window_s):
    """Rolling Pearson correlation between residuals of physically-linked sensors."""
    ts = pd.DatetimeIndex(pd.to_datetime(df_wide["timestamp"], utc=True))
    w = f"{int(round(window_s))}s"
    out = {}
    for a, b in CORRELATION_PAIRS:
        ca, cb = f"{a}__residual", f"{b}__residual"
        if ca not in df_wide or cb not in df_wide:
            continue
        sa = pd.Series(df_wide[ca].to_numpy(float), index=ts)
        sb = pd.Series(df_wide[cb].to_numpy(float), index=ts)
        corr = sa.rolling(w, min_periods=pm.MIN_ROLL_SAMPLES).corr(sb)
        out[f"corr__{a}__{b}"] = corr.fillna(0.0).to_numpy()
    return out


def build_ml_features(long_path="engine_residuals.csv", wide_path="engine_residual_features.csv"):
    long_df = pd.read_csv(long_path, parse_dates=["timestamp"])
    wide_df = pd.read_csv(wide_path, parse_dates=["timestamp"])

    long_df = _rate_of_change(long_df)
    long_df["pct_error"] = _pct_error(long_df)

    # fold the two new per-sensor features into the wide table
    for feat in ("rate_of_change", "pct_error"):
        piv = long_df.pivot(index="timestamp", columns="sensor", values=feat)
        piv.columns = [f"{c}__{feat}" for c in piv.columns]
        wide_df = wide_df.merge(piv.reset_index(), on="timestamp", how="left")

    corr_feats = _cross_correlations(wide_df, pm.ROLL_WINDOW_S)
    for name, values in corr_feats.items():
        wide_df[name] = values

    wide_df.to_csv("ml_features.csv", index=False)
    return wide_df


# columns that must never be used as ML inputs (ground truth / identifiers)
NON_FEATURE_COLUMNS = {
    "timestamp", "gt_operating_condition", "gt_fault_type", "gt_fault_severity",
    "top_sensor",
}


def feature_columns(df):
    """All numeric columns usable as ML input (excludes ground truth/IDs)."""
    return [c for c in df.columns
            if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]


if __name__ == "__main__":
    df = build_ml_features()
    print(f"ml_features.csv written: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"{len(feature_columns(df))} usable ML feature columns")
