"""All CSV column knowledge lives here. Nothing else in the dashboard hard-codes names."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # the "Digital Twin/" folder
RESULTS_CSV = ROOT / "engine_health_results.csv"   # pipeline output (read-only)
RAW_CSV = ROOT / "engine_dataset.csv"              # raw telemetry (actual sensor values)

TIME_COL = "timestamp"

# key = column prefix used in engine_health_results.csv
SENSORS = {
    "cht_c":              {"label": "CHT",             "unit": "°C"},
    "egt_c":              {"label": "EGT",             "unit": "°C"},
    "oil_temperature_c":  {"label": "Oil Temperature", "unit": "°C"},
    "oil_pressure_bar":   {"label": "Oil Pressure",    "unit": "bar"},
    "fuel_flow_l_h":      {"label": "Fuel Flow",       "unit": "L/h"},
    "fuel_pressure_bar":  {"label": "Fuel Pressure",   "unit": "bar"},
    "vibration_g":        {"label": "Vibration",       "unit": "g"},
    "voltage_v":          {"label": "Voltage",         "unit": "V"},
}

# per-sensor suffixes (column = f"{sensor}__{suffix}")
SENSOR_FIELDS = ["residual", "z", "anomaly_score", "status_level",
                 "rate_of_change", "pct_error", "roll_mean_abs_z", "persist_s"]

# engine-level columns
ENGINE_FIELDS = {
    "in_transient": "in_transient",
    "engine_max_score": "engine_max_score",
    "engine_max_status_level": "engine_max_status_level",
    "n_watch": "n_sensors_watch_or_worse",
    "n_anomalous": "n_sensors_anomalous_or_worse",
    "top_sensor": "top_sensor",
    "operating_condition": "gt_operating_condition",   # ground truth from simulator
    "gt_fault_type": "gt_fault_type",
    "gt_fault_severity": "gt_fault_severity",
    "fault_category": "fault_category",
    "predicted_fault": "predicted_fault_category",
    "predicted_confidence": "predicted_fault_confidence",
    "ml_score": "ml_anomaly_score",
    "ml_label": "ml_anomaly_label",
    "health_index": "health_index",
    "health_status": "health_status",
    "health_components": ["health_component_ml", "health_component_stage3",
                          "health_component_persistence", "health_component_breadth"],
}

# residual = actual - expected  (+1)  or  expected - actual (-1).
# CONFIRMED +1 (fault-direction test: overheating -> CHT/EGT residual > 0, low oil pressure -> < 0).
RESIDUAL_SIGN = 1

MAX_POINTS_DEFAULT = 1500   # server-side downsampling cap per series request

# Display names (inferred from column names; edit freely)
STATUS_LEVEL_NAMES = {"0": "NORMAL", "1": "WATCH", "2": "ANOMALOUS", "3": "LEVEL 3"}
FAULT_DISPLAY = {
    "normal_operation": "Normal operation",
    "combustion_misfire": "Combustion misfire",
    "other_unmapped": "Unclassified anomaly",
    "normal": "None (normal)", "misfire": "Misfire", "overheating": "Overheating",
    "low_oil_pressure": "Low oil pressure", "high_vibration": "High vibration",
}
