"""
health_index.py  --  STAGE 4d: interpretable Engine Health Index (0-100)

Combines evidence that already exists at this point in the pipeline into a
single explainable number. Nothing here is a new measurement -- it is a
documented, configurable weighted combination of:

    1. ml_anomaly_score        (0-100, from Isolation Forest)         weight W_ML
    2. engine_max_score         (0-100, worst per-sensor Stage-3 score) weight W_STAGE3
    3. persistence_component    (0-100, from Stage-3 persist_s, worst sensor) weight W_PERSIST
    4. n_sensors_anomalous_or_worse (count -> 0-100 scaled)            weight W_BREADTH

health_index = 100 - clip(weighted degradation, 0, 100)

WHY THESE FOUR AND THESE WEIGHTS
---------------------------------
- ml_anomaly_score carries the multivariate ML view (heaviest weight: it is
  the most information-dense signal).
- engine_max_score re-uses the physics-grounded Stage-3 single-sensor score,
  so the index does not depend on the ML model alone.
- persistence penalizes intermittent noise vs sustained abnormal behaviour,
  directly answering "how long has this been going on".
- breadth (how many sensors are simultaneously off) distinguishes a
  localized sensor issue from a systemic engine problem.

These weights are PROTOTYPE ASSUMPTIONS (configurable constants below), not
a validated aviation health formula.

TRANSIENT HANDLING
-------------------
anomaly_detection.py leaves ml_anomaly_score as NaN ("not evaluated") during
a start-up/shutdown transient, since the ML model was never trained to judge
those rows. NaN would otherwise poison the weighted sum (NaN + anything =
NaN), silently blanking the health index during every start-up and
shutdown. Instead: the NaN is treated as "no ML evidence either way" (0
contribution to degradation) for the purpose of computing a number, but the
row's status is overridden to "TRANSIENT" so the dashboard shows "starting
up / shutting down" rather than presenting a HEALTHY/WARNING/ALERT verdict
the system isn't in a position to make yet.
"""

import numpy as np
import pandas as pd

W_ML, W_STAGE3, W_PERSIST, W_BREADTH = 0.40, 0.30, 0.20, 0.10   # sum = 1.0
PERSIST_SAT_S = 60.0     # seconds of persistence that saturates the persistence component
BREADTH_SAT_N = 4        # sensors simultaneously anomalous-or-worse that saturates breadth

WARNING_BELOW, ALERT_BELOW = 70.0, 40.0   # health_index thresholds -- prototype assumptions


def _persist_component(df):
    persist_cols = [c for c in df.columns if c.endswith("__persist_s")]
    if not persist_cols:
        return np.zeros(len(df))
    worst = df[persist_cols].max(axis=1).to_numpy(float)
    return np.clip(worst / PERSIST_SAT_S, 0.0, 1.0) * 100.0


def compute_health_index(df):
    ml_raw = df.get("ml_anomaly_score", pd.Series(0.0, index=df.index)).to_numpy(float)
    ml_is_unknown = np.isnan(ml_raw)                # transient rows: "not evaluated"
    ml = np.nan_to_num(ml_raw, nan=0.0)              # 0 contribution to degradation, not NaN
    stage3 = df.get("engine_max_score", pd.Series(0.0, index=df.index)).to_numpy(float)
    persist = _persist_component(df)
    breadth = np.clip(
        df.get("n_sensors_anomalous_or_worse", pd.Series(0, index=df.index)).to_numpy(float)
        / BREADTH_SAT_N, 0.0, 1.0) * 100.0

    degradation = W_ML * ml + W_STAGE3 * stage3 + W_PERSIST * persist + W_BREADTH * breadth
    health = np.clip(100.0 - degradation, 0.0, 100.0)

    status = np.select(
        [health < ALERT_BELOW, health < WARNING_BELOW],
        ["ALERT", "WARNING"], default="HEALTHY").astype(object)

    # in_transient (if present) is the authoritative signal; ml_is_unknown is a
    # fallback in case in_transient wasn't carried through to this file's input.
    in_transient = df["in_transient"].astype(bool).to_numpy() if "in_transient" in df else ml_is_unknown
    status[in_transient] = "TRANSIENT"

    out = df.copy()
    out["health_index"] = health
    out["health_status"] = status
    out["health_component_ml"] = ml_raw   # kept as NaN where unknown, for transparency
    out["health_component_stage3"] = stage3
    out["health_component_persistence"] = persist
    out["health_component_breadth"] = breadth
    return out


def explain_row(row):
    """Human-readable contributing-factor summary for one row (dashboard use)."""
    if row["health_status"] == "TRANSIENT":
        return ("Engine Health: not evaluated (TRANSIENT). The engine is starting up or "
                "shutting down, so a health verdict is withheld until it stabilizes.")

    comps = {
        "ML anomaly": row["health_component_ml"],
        "Stage-3 worst-sensor score": row["health_component_stage3"],
        "Persistence": row["health_component_persistence"],
        "Breadth (sensors affected)": row["health_component_breadth"],
    }
    # health_component_ml may still be NaN here in edge cases; drop it from display rather
    # than let "nan" print in the sentence.
    comps = {k: v for k, v in comps.items() if np.isfinite(v)}
    top = sorted(comps.items(), key=lambda kv: -kv[1])[:2]
    parts = [f"{name} ({val:.0f}/100)" for name, val in top if val > 0]
    reason = " and ".join(parts) if parts else "no significant deviation"
    top_sensor = row.get("top_sensor", "n/a")
    fault = row.get("predicted_fault_category", "n/a")
    return (f"Engine Health: {row['health_index']:.0f}/100 ({row['health_status']}). "
            f"Main contributor(s): {reason}. Most-affected sensor: {top_sensor}. "
            f"Probable category (prototype, synthetic-trained): {fault}.")


def run(path="ml_classification_results.csv"):
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
    except FileNotFoundError:
        df = pd.read_csv("ml_anomaly_results.csv", parse_dates=["timestamp"])
        print("No classification results found; running health index on anomaly-only output.")

    out = compute_health_index(df)
    out.to_csv("engine_health_results.csv", index=False)

    print("\nHealth index summary:")
    print(out["health_status"].value_counts().to_string())

    evaluated = out[out["health_status"] != "TRANSIENT"]
    if len(evaluated):
        worst = evaluated.loc[evaluated["health_index"].idxmin()]
        print("\nWorst EVALUATED row in this run (transients excluded):")
        print(explain_row(worst))
    else:
        print("\nEvery row in this run was TRANSIENT -- no evaluated rows to report on.")
    return out


if __name__ == "__main__":
    run()