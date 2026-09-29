"""Data access layer. The dashboard only talks to a DataSource, so a live
pipeline can replace CsvSource later without touching the frontend."""
import numpy as np
import pandas as pd
import config as C


def _clean(v):
    """Make a value JSON-safe (NaN/inf -> None, numpy -> python)."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if pd.isna(v):
        return None
    return v


def _find_col(df, name):
    if name in df.columns:
        return name
    low = {c.lower(): c for c in df.columns}
    return low.get(name.lower())


class CsvSource:
    def __init__(self, results_path=C.RESULTS_CSV, raw_path=C.RAW_CSV):
        self.results_path, self.raw_path = results_path, raw_path
        self.report = {"merge": None, "missing_columns": [], "raw_sensor_columns": {}}
        self.df = self._load()

    # ---------- loading ----------
    def _load(self):
        df = pd.read_csv(self.results_path)
        tcol = C.TIME_COL
        parsed = pd.to_datetime(df[tcol], errors="coerce") if tcol in df.columns else None
        if parsed is not None and parsed.notna().mean() > 0.9:
            df[tcol] = parsed
            self.time_kind = "datetime"
        else:
            self.time_kind = "numeric"
        df = self._merge_raw(df)
        self._check_columns(df)
        return df.reset_index(drop=True)

    def _merge_raw(self, df):
        try:
            raw = pd.read_csv(self.raw_path)
        except FileNotFoundError:
            self.report["merge"] = "raw dataset not found - actual/expected unavailable"
            return df
        tcol = C.TIME_COL
        if tcol in raw.columns and self.time_kind == "datetime":
            raw[tcol] = pd.to_datetime(raw[tcol], errors="coerce")
        cols = {}
        for key in C.SENSORS:
            rc = _find_col(raw, key)
            if rc:
                cols[key] = rc
        self.report["raw_sensor_columns"] = cols
        if not cols:
            self.report["merge"] = "no matching sensor columns in raw dataset"
            return df
        sub = raw[[c for c in cols.values()]].copy()
        sub.columns = [f"{k}__actual" for k in cols]
        if tcol in raw.columns:
            sub[tcol] = raw[tcol]
            out = df.merge(sub.drop_duplicates(tcol), on=tcol, how="left")
            self.report["merge"] = f"merged on '{tcol}'"
        elif len(raw) == len(df):
            out = pd.concat([df, sub.reset_index(drop=True)], axis=1)
            self.report["merge"] = "aligned by row position (same length)"
        else:
            self.report["merge"] = "could not align raw dataset"
            return df
        exp = {f"{k}__expected": out[f"{k}__actual"] - C.RESIDUAL_SIGN * out[f"{k}__residual"]
               for k in cols}
        return pd.concat([out, pd.DataFrame(exp, index=out.index)], axis=1)

    def _check_columns(self, df):
        want = [C.TIME_COL] + [v for k, v in C.ENGINE_FIELDS.items() if isinstance(v, str)]
        want += [c for k in ("health_components",) for c in C.ENGINE_FIELDS[k]]
        want += [f"{s}__{f}" for s in C.SENSORS for f in ("residual", "z", "status_level")]
        self.report["missing_columns"] = [c for c in want if c not in df.columns]

    # ---------- helpers ----------
    def _tval(self, v):
        return _clean(v)

    def _col(self, name):
        return name if name in self.df.columns else None

    # ---------- queries ----------
    def meta(self):
        t = self.df[C.TIME_COL]
        alert = self.df.index[self.df["health_status"] == "ALERT"] if "health_status" in self.df else []
        return {
            "dt_seconds": self._dt(),
            "default_index": int(alert[-1]) if len(alert) else len(self.df) - 1,
            "labels": {"status_levels": C.STATUS_LEVEL_NAMES, "faults": C.FAULT_DISPLAY},
            "rows": len(self.df),
            "time_kind": self.time_kind,
            "start": _clean(t.iloc[0]), "end": _clean(t.iloc[-1]),
            "sensors": [{"key": k, **v, "has_actual": f"{k}__actual" in self.df.columns}
                        for k, v in C.SENSORS.items()],
            "source_mode": "simulated telemetry (CSV playback)",
            "report": self.report,
        }

    def schema(self):
        cats = {}
        for key in ["health_status", "predicted_fault", "gt_fault_type", "fault_category",
                    "operating_condition", "top_sensor", "ml_label", "engine_max_status_level"]:
            col = self._col(C.ENGINE_FIELDS[key])
            if col:
                vc = self.df[col].value_counts(dropna=False).head(20)
                cats[key] = {str(k): int(v) for k, v in vc.items()}
        lv = {}
        for s in C.SENSORS:
            col = self._col(f"{s}__status_level")
            if col:
                lv[s] = sorted({str(x) for x in self.df[col].dropna().unique()})[:10]
        return {"categorical_values": cats, "sensor_status_levels": lv,
                "null_counts": {c: int(n) for c, n in self.df.isna().sum().items() if n > 0}}

    def latest(self, index=None):
        i = len(self.df) - 1 if index is None else max(0, min(int(index), len(self.df) - 1))
        r = self.df.iloc[i]
        eng = {k: _clean(r[v]) for k, v in C.ENGINE_FIELDS.items()
               if isinstance(v, str) and v in self.df.columns}
        eng["health_components"] = {c: _clean(r[c]) for c in C.ENGINE_FIELDS["health_components"]
                                    if c in self.df.columns}
        sensors = {}
        for k, m in C.SENSORS.items():
            g = lambda f: _clean(r.get(f"{k}__{f}"))
            sensors[k] = {**m, "actual": g("actual"), "expected": g("expected"),
                          "residual": g("residual"), "z": g("z"),
                          "anomaly_score": g("anomaly_score"), "status_level": g("status_level"),
                          "rate_of_change": g("rate_of_change")}
        return {"index": i, "timestamp": _clean(r[C.TIME_COL]), "engine": eng, "sensors": sensors}

    def _dt(self):
        if self.time_kind != "datetime":
            return None
        d = self.df[C.TIME_COL].diff().dt.total_seconds().median()
        return None if pd.isna(d) else float(d)

    def _pick(self, start, end, max_points, score=None):
        """Row indices for [start,end). If too many, keep the 'worst' row per bucket
        (so short anomalies/dips are not lost by downsampling)."""
        n = len(self.df)
        start = max(0, int(start)); end = n if end is None else min(int(end), n)
        idx = np.arange(start, max(start, end))
        max_points = max(50, min(int(max_points), 5000))
        if len(idx) <= max_points:
            return idx
        chunks = np.array_split(idx, max_points)
        if score is None:
            return np.array([c[len(c) // 2] for c in chunks])
        return np.array([c[int(np.argmax(score[c]))] for c in chunks])

    def series(self, sensor, start=0, end=None, max_points=C.MAX_POINTS_DEFAULT):
        if sensor not in C.SENSORS:
            raise KeyError(sensor)
        zc = f"{sensor}__z"
        score = np.nan_to_num(self.df[zc].abs().to_numpy(dtype=float)) if zc in self.df else None
        idx = self._pick(start, end, max_points, score)
        cols = [C.TIME_COL] + [f"{sensor}__{f}" for f in ("actual", "expected", "residual", "z")
                               if f"{sensor}__{f}" in self.df.columns]
        sub = self.df.iloc[idx][cols]
        out = {"sensor": sensor, **C.SENSORS[sensor], "index": idx.tolist(),
               "t": [_clean(x) for x in sub[C.TIME_COL]]}
        for c in cols[1:]:
            out[c.split("__")[1]] = [_clean(x) for x in sub[c]]
        return out

    def health_series(self, start=0, end=None, max_points=C.MAX_POINTS_DEFAULT):
        hi = C.ENGINE_FIELDS["health_index"]
        score = 100 - np.nan_to_num(self.df[hi].to_numpy(dtype=float), nan=100.0)
        idx = self._pick(start, end, max_points, score)
        cols = [c for c in (C.TIME_COL, hi, C.ENGINE_FIELDS["health_status"],
                            C.ENGINE_FIELDS["in_transient"]) if c in self.df.columns]
        sub = self.df.iloc[idx][cols]
        return {"index": idx.tolist(), "t": [_clean(x) for x in sub[C.TIME_COL]],
                "health_index": [_clean(x) for x in sub[hi]],
                "health_status": [_clean(x) for x in sub[C.ENGINE_FIELDS["health_status"]]]}
