"""
pipeline.py  --  orchestrates the existing Stage 3 output through the new Stage 4 modules.

BATCH MODE (this file, as run below)
    python pipeline.py engine_dataset.csv
    1. physics_model.py            (UNCHANGED)  -> engine_residual_features.csv etc.
    2. ml_features.py                            -> ml_features.csv
    3. anomaly_detection.py                      -> ml_anomaly_results.csv, anomaly_model.joblib
    4. fault_classification.py                   -> ml_classification_results.csv, fault_classifier.joblib
    5. health_index.py                           -> engine_health_results.csv

REAL-TIME / STREAMING NOTE
---------------------------
This batch script recomputes rolling windows over the whole file, which is
correct for training and for offline verification. For true real-time use
without restarting the app, physics_model.py's rolling/trend/persistence
logic already operates on a growing time-indexed buffer (pandas .rolling on
a DatetimeIndex), so the natural extension is:

    class LiveScorer:
        def __init__(self, twin, anomaly_model, fault_model):
            self.buffer = pd.DataFrame()   # keep e.g. last TREND_WINDOW_S of rows
        def push(self, new_row):
            self.buffer = pd.concat([self.buffer, new_row]).last("...s")
            pred = self.twin.predict(self.buffer)
            long, wide = residual_analysis(self.buffer, pred)   # cheap: small buffer
            feats = build_ml_features_from(wide)
            ... score just the LAST row with the already-fitted models ...

The anomaly and fault models are trained once (offline, this pipeline) and
then only used with .score()/.predict() online -- they do not need to be
retrained per observation. This module stops at the batch/offline stage, as
requested; RUL and the live buffer class are later-stage work.

RUL (STAGE 5, NOT IMPLEMENTED YET)
------------------------------------
Left as a placeholder function below. The intended approach: fit a simple
degradation-trend model (e.g. linear or exponential extrapolation) on the
health_index time series *per active fault category*, project forward to a
configurable failure threshold (e.g. health_index == ALERT_BELOW), and
report the estimated remaining time as an EXPERIMENTAL / SIMULATED value --
explicitly not an aviation-certified maintenance prediction. This requires
either multiple synthetic run-to-failure datasets or a documented synthetic
degradation generator, which does not exist yet and should be built as its
own labelled task before RUL is implemented.
"""

import sys
import physics_model as pm
import ml_features as mf
import anomaly_detection as ad
import fault_classification as fc
import health_index as hi


def estimate_rul(*args, **kwargs):
    raise NotImplementedError(
        "RUL is a later stage (Stage 5). It requires a documented synthetic "
        "run-to-failure dataset before it can be trained; see module docstring."
    )


def run_pipeline(csv_path="engine_dataset.csv"):
    # ---- Stage 2/3: existing, unchanged ----
    df = pm.load_dataset(csv_path)
    twin = pm.EngineTwin().calibrate(df)
    pred = twin.predict(df)
    long, wide = pm.residual_analysis(df, pred)
    long.to_csv("engine_residuals.csv", index=False)
    wide.to_csv("engine_residual_features.csv", index=False)
    pm.plot_residual_analysis(long)
    pm.print_verification(long, wide)

    # ---- Stage 4: new ML layer ----
    ml_df = mf.build_ml_features()
    anomaly_df, anomaly_model = ad.run()
    class_df, fault_model = fc.run()
    health_df = hi.run()

    print("\nPipeline complete. Final table: engine_health_results.csv")
    return health_df


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "engine_dataset.csv"
    run_pipeline(path)
