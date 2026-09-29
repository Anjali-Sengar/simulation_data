"""
fault_classification.py  --  STAGE 4c: prototype fault classification

WHY THIS COMES AFTER ANOMALY DETECTION
---------------------------------------
The anomaly detector (anomaly_detection.py) answers "is something wrong?"
without needing labels. This module answers "what KIND of problem does it
look like?" -- and for that we DO need example patterns of each fault kind
to learn from. Because no certified/real aviation fault database exists for
this prototype, the classifier is trained on your SYNTHETIC fault_type
labels (fault_type column from the data generator). This is explicitly a
PROTOTYPE / SYNTHETIC-LABEL classifier, not a certified diagnostic model.

FAULT CATEGORY MAPPING
-----------------------
Your generator's fault_type values are mapped onto the prototype category
list from the spec. Update FAULT_CATEGORY_MAP if your generator uses
different fault_type strings -- unmapped values fall into "other/unmapped"
so nothing is silently dropped.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import ml_features as mf

RANDOM_STATE = 42

# --- prototype fault categories (spec Step 3) -> map your generator's fault_type strings here
FAULT_CATEGORY_MAP = {
    "normal":            "normal_operation",
    "overheat":          "overheating_thermal",
    "lube":              "lubrication_oil_pressure",
    "fuel":              "injector_fuel_flow",
    "misfire":           "combustion_misfire",
    "vibration":         "mechanical_vibration",
    "sensor_drift":      "sensor_drift_failure",
}
DEFAULT_CATEGORY = "other_unmapped"


class FaultClassifier:
    def __init__(self, n_estimators=300, random_state=RANDOM_STATE):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state,
            class_weight="balanced_subsample", max_depth=None,
        )
        self.feature_names_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.feature_names_ = list(X.columns)
        Xn = np.nan_to_num(X.to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
        self.model.fit(Xn, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        Xn = np.nan_to_num(X[self.feature_names_].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
        proba = self.model.predict_proba(Xn)
        pred = self.classes_[proba.argmax(axis=1)]
        conf = proba.max(axis=1)
        return pred, conf, proba

    def feature_importance(self, top=10):
        s = pd.Series(self.model.feature_importances_, index=self.feature_names_)
        return s.sort_values(ascending=False).head(top)

    def save(self, path="fault_classifier.joblib"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="fault_classifier.joblib"):
        return joblib.load(path)


def run(ml_results_path="ml_anomaly_results.csv"):
    df = pd.read_csv(ml_results_path, parse_dates=["timestamp"])
    if "gt_fault_type" not in df:
        print("No fault_type labels available -- skipping classification "
              "(anomaly detection output stands alone).")
        return None, None

    df["fault_category"] = df["gt_fault_type"].map(FAULT_CATEGORY_MAP).fillna(DEFAULT_CATEGORY)
    feats = mf.feature_columns(df)
    X, y = df[feats], df["fault_category"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE,
        stratify=y if y.value_counts().min() >= 2 else None)

    clf = FaultClassifier().fit(Xtr, ytr)
    pred, conf, _ = clf.predict(Xte)

    print("\nFault classification report (held-out synthetic test rows):")
    print(classification_report(yte, pred, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    labels = sorted(y.unique())
    print(pd.DataFrame(confusion_matrix(yte, pred, labels=labels), index=labels, columns=labels))
    print("\nTop features driving the classifier:")
    print(clf.feature_importance().round(4).to_string())

    # score the WHOLE dataset (so downstream health index has a label per row)
    pred_all, conf_all, _ = clf.predict(X)
    df["predicted_fault_category"] = pred_all
    df["predicted_fault_confidence"] = conf_all
    df.to_csv("ml_classification_results.csv", index=False)
    clf.save()
    return df, clf


if __name__ == "__main__":
    run()
