"""
anomaly_detection.py  --  STAGE 4b: ML anomaly detection

WHY ISOLATION FOREST
--------------------
- Unsupervised: trains only on rows we can trust as healthy (gt_fault_type ==
  "normal" and not in a start-up/RPM transient), so it does not need real
  labelled aviation fault data.
- Learns MULTIVARIATE structure: it can catch a combination of small,
  individually-unremarkable residual shifts across several sensors at once,
  which the single-sensor Stage-3 z-scores cannot see.
- Cheap to train/tune and gives a continuous anomaly score (not just a
  label), which the Health Index module needs.
- Reasonable, explainable choice for a student prototype; an autoencoder or
  One-Class SVM would add tuning complexity without a clear benefit here.

The model NEVER sees gt_fault_type, gt_operating_condition, gt_fault_severity,
or timestamp as inputs -- only the Stage-3 + Stage-4a engineered features.

TRANSIENT HANDLING
-------------------
The model is trained only on stable healthy rows (in_transient == False), so
it never learns what normal start-up/shutdown feature patterns look like.
Scoring a transient row with it therefore isn't a real health judgement --
it's asking the model to grade something it was never taught. So at scoring
time, any row flagged in_transient is labelled "TRANSIENT" instead of
NORMAL/ANOMALOUS, the same way a real monitoring system suppresses
diagnostics during a known start-up/shutdown window rather than issuing a
false fault call.

OUTPUT
------
adds to the wide feature table:
    ml_raw_score      : IsolationForest decision_function (higher = more normal)
    ml_anomaly_score  : rescaled to 0-100, higher = MORE anomalous (matches
                         the convention already used by Stage 3's anomaly_score).
                         NaN for rows in a transient window (not a meaningful score).
    ml_anomaly_label  : "NORMAL" / "ANOMALOUS" / "TRANSIENT". NORMAL/ANOMALOUS come
                         from a threshold calibrated on the healthy training
                         distribution (a percentile of its own scores), NOT an
                         arbitrary fixed threshold. TRANSIENT overrides both when
                         in_transient is True, since the model has no basis to
                         judge those rows.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import ml_features as mf

RANDOM_STATE = 42
CONTAMINATION_TRAINING_PERCENTILE = 1.0   # threshold = 1st percentile of HEALTHY scores
                                            # (i.e. ~1% false-alarm rate on healthy data
                                            #  by construction) -- prototype assumption,
                                            #  tune for your data.


class ResidualAnomalyDetector:
    def __init__(self, n_estimators=200, random_state=RANDOM_STATE):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators, random_state=random_state,
            contamination="auto",   # we calibrate our own threshold below instead
        )
        self.feature_names_ = None
        self.threshold_ = None      # raw decision_function threshold
        self.score_min_ = None      # for 0-100 rescaling
        self.score_max_ = None

    def fit(self, wide_df, feature_cols):
        healthy = (wide_df.get("gt_fault_type", pd.Series("normal", index=wide_df.index)) == "normal")
        if "in_transient" in wide_df:
            healthy = healthy & ~wide_df["in_transient"].astype(bool)
        if healthy.sum() < 30:
            healthy = (wide_df.get("gt_fault_type", pd.Series("normal", index=wide_df.index)) == "normal")

        X = wide_df.loc[healthy, feature_cols].to_numpy(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.feature_names_ = list(feature_cols)

        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)

        healthy_scores = self.model.decision_function(Xs)   # higher = more normal
        self.threshold_ = float(np.percentile(healthy_scores, CONTAMINATION_TRAINING_PERCENTILE))
        self.score_min_ = float(healthy_scores.min())
        self.score_max_ = float(healthy_scores.max())
        print(f"IsolationForest trained on {healthy.sum()} healthy, non-transient rows, "
              f"{len(feature_cols)} features. Threshold (raw)={self.threshold_:.4f}")
        return self

    def score(self, wide_df):
        """
        wide_df must contain the feature columns; an 'in_transient' column is
        used if present to mark rows the model was never trained to judge.
        """
        X = wide_df[self.feature_names_].to_numpy(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = self.scaler.transform(X)
        raw = self.model.decision_function(Xs)   # higher = more normal

        # rescale so that HIGHER = MORE ANOMALOUS, 0-100, anchored on the
        # healthy training distribution (not the current batch's own range,
        # so scores are comparable run-to-run).
        span = max(self.score_max_ - self.score_min_, 1e-6)
        normal_ness = np.clip((raw - self.score_min_) / span, 0.0, 1.0)
        anomaly_score = 100.0 * (1.0 - normal_ness)
        label = np.where(raw < self.threshold_, "ANOMALOUS", "NORMAL").astype(object)

        if "in_transient" in wide_df:
            in_transient = wide_df["in_transient"].astype(bool).to_numpy()
            label[in_transient] = "TRANSIENT"
            anomaly_score = anomaly_score.astype(float)
            anomaly_score[in_transient] = np.nan   # not a meaningful health score

        return raw, anomaly_score, label

    def save(self, path="anomaly_model.joblib"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="anomaly_model.joblib"):
        return joblib.load(path)


def run(ml_features_path="ml_features.csv"):
    df = pd.read_csv(ml_features_path, parse_dates=["timestamp"])
    feats = mf.feature_columns(df)

    det = ResidualAnomalyDetector().fit(df, feats)
    raw, score, label = det.score(df)
    df["ml_raw_score"] = raw
    df["ml_anomaly_score"] = score
    df["ml_anomaly_label"] = label

    df.to_csv("ml_anomaly_results.csv", index=False)
    det.save()

    if "gt_fault_type" in df:
        healthy = df["gt_fault_type"] == "normal"
        stable = ~df["in_transient"].astype(bool) if "in_transient" in df else pd.Series(True, index=df.index)
        print("\nAnomaly-detector verification:")
        print(f"  Healthy, STABLE rows flagged ANOMALOUS (false alarms): "
              f"{(label[(healthy & stable).to_numpy()] == 'ANOMALOUS').mean() * 100:.1f}%")
        print(f"  Faulty, STABLE rows flagged ANOMALOUS (detections):    "
              f"{(label[(~healthy & stable).to_numpy()] == 'ANOMALOUS').mean() * 100:.1f}%")
        if "in_transient" in df:
            n_trans = int(df["in_transient"].astype(bool).sum())
            print(f"  Rows labelled TRANSIENT (excluded from the above, not scored): {n_trans}")
    return df, det


if __name__ == "__main__":
    run()