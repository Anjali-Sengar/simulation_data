"""Flask backend. Read-only w.r.t. the Digital Twin pipeline."""
from flask import Flask, jsonify, request, render_template
from data_source import CsvSource
import config as C

app = Flask(__name__)
source = CsvSource()          # swap for a live source in Stage 6


@app.get("/")
def index():
    return render_template("dashboard.html")     # added in Stage 2


@app.get("/api/meta")
def meta():
    return jsonify(source.meta())


@app.get("/api/schema")
def schema():
    return jsonify(source.schema())


@app.get("/api/latest")
def latest():
    return jsonify(source.latest(request.args.get("index")))


@app.get("/api/series")
def series():
    try:
        return jsonify(source.series(
            request.args.get("sensor", ""),
            request.args.get("start", 0),
            request.args.get("end"),
            int(request.args.get("max_points", C.MAX_POINTS_DEFAULT))))
    except KeyError:
        return jsonify({"error": "unknown sensor", "valid": list(C.SENSORS)}), 400


@app.get("/api/health_series")
def health_series():
    return jsonify(source.health_series(
        request.args.get("start", 0), request.args.get("end"),
        int(request.args.get("max_points", C.MAX_POINTS_DEFAULT))))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
