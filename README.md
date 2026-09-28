# Aero Piston Engine Synthetic Telemetry Generator

A continuous, physically-motivated synthetic data generator for **one**
virtual 4-cylinder, 4-stroke, horizontally-opposed, air-cooled,
gasoline-powered aero piston engine, intended as labeled/unlabeled input
data for a downstream MALE UAV engine health-monitoring ML pipeline.

This tool **only generates data**. It does not run any ML model, anomaly
detector, health-score calculation, or dashboard.

---

## 1. How the engine is modeled

The engine represented is loosely based on the ~115–160 hp class of
direct-drive, horizontally-opposed, air-cooled 4-cylinder aero engines
(e.g. Lycoming/Continental O-235/O-320 family), which is representative of
the piston powerplants used on many MALE-class UAVs. All numeric ranges
below are engineering approximations for realism, not a spec of any single
certified engine — tune `EngineConfig` in `aero_engine_simulator.py` to
match a specific real engine if needed.

### State-machine approach (not independent random sampling)

Every parameter is a **stateful variable**. At each 1-second tick:

1. A **target value** is computed from the current throttle, RPM, engine
   load, altitude/ambient conditions, and any active fault.
2. The parameter moves toward that target using first-order exponential
   smoothing: `value += (target - value) * (1 - exp(-dt / tau))`, where
   `tau` (seconds) controls how fast that specific parameter can change.
3. Small Gaussian sensor noise is added on top.

This guarantees:
- No independent/random values per timestamp — everything is continuous.
- Different thermal masses behave differently: EGT (tau = 20 s) reacts
  fastest, CHT (tau = 120 s) reacts more slowly, and oil temperature
  (tau = 240 s) reacts slowest of all, matching real engine thermal
  behavior.
- No sudden unrealistic jumps — even a full throttle chop moves RPM,
  load, and temperatures smoothly toward their new targets.

### Causal relationships enforced

- **RPM** is derived from throttle position: `rpm_target = idle_rpm +
  (max_rpm - idle_rpm) * throttle_fraction`.
- **Engine load** is derived from a blend of RPM and throttle fraction,
  scaled by air density (so load capability falls off with altitude):
  `load = 100 * (0.30*rpm_frac + 0.70*throttle_frac) * density_ratio^0.6`.
- **CHT and EGT** are derived from ambient temperature + engine load
  (higher load → hotter). EGT responds quickly; CHT responds slowly.
- **Oil temperature** rises with load but on a very slow (several-minute)
  time constant, and in turn **oil pressure** falls slightly as oil
  temperature rises (thinner oil), while rising with RPM (more pump
  speed).
- **Fuel flow** scales with engine load; **fuel pressure** is mostly
  regulated but drifts slightly with RPM.
- **Vibration** has a small baseline that scales gently with RPM/load and
  stays low/stable in normal operation.
- **Voltage** reflects battery-only operation before start and during
  cranking (~9.5 V), then alternator-regulated 13.8–14.4 V once running,
  scaling slightly with RPM.
- **Ambient temperature/pressure** are derived from a simulated altitude
  (via the ISA standard-atmosphere model) plus a slow random-walk
  "weather" drift, so they only ever change gradually — consistent with
  climb/cruise/descent altitude changes.

### Parameter ranges used (approximate, at sea level / cfg defaults)

| Parameter              | Idle      | Cruise     | Takeoff/Max | Notes |
|-------------------------|-----------|------------|-------------|-------|
| RPM                     | ~650      | ~2350      | ~2700       | direct-drive, redline 2700 |
| Throttle (%)            | ~8        | ~65        | 100         | |
| Engine load (%)         | ~10–15    | ~70        | 100         | |
| CHT (°C)                | ~120–135  | ~175–190   | ~200–215    | ceiling clipped at 235 °C |
| EGT (°C)                | ~350–420  | ~680–740   | ~800–850    | ceiling clipped at 900 °C |
| Oil pressure (bar)      | ~2.0      | ~4.5–5.0   | ~5.5–6.0    | |
| Oil temperature (°C)    | ~65–75    | ~90–100    | ~100–110    | ceiling clipped at 125 °C |
| Fuel flow (L/h)         | ~9–10     | ~30–35     | ~45–50      | |
| Fuel pressure (bar)     | ~0.33     | ~0.42      | ~0.45       | |
| Vibration (g)           | ~0.06–0.08| ~0.10–0.12 | ~0.13       | normal operation |
| Voltage (V)             | 13.8      | ~14.0–14.2 | 14.2–14.4   | 9.5 V while cranking |
| Ambient temp (°C)       | ISA(altitude) + slow weather drift, sea-level default 15 °C |
| Ambient pressure (hPa)  | ISA(altitude) + slow weather drift, sea-level default 1013.25 hPa |

All of these are configurable via the `EngineConfig` dataclass at the top
of `aero_engine_simulator.py` (base offsets, load coefficients, time
constants `tau_*`, and noise sigmas `noise_*` are all named fields).

---

## 2. Operating conditions (mission profile)

The default mission is a full flight cycle, built by `build_default_mission()`:

```
start (20s) → idle (60s) → taxi (90s) → takeoff (60s) → climb (300s)
  → cruise (600s, repeatable) → descent (300s) → taxi (60s) → shutdown (20s)
```

- **start**: simulates starter cranking (RPM ramps 0 → ~300, voltage dips
  to ~9.5 V), then the engine "catches" and RPM climbs smoothly to idle.
- **idle / taxi**: low, steady throttle.
- **takeoff**: throttle ramps quickly to 100%.
- **climb**: throttle eases slightly from 100% to ~85%; altitude increases
  at a configurable climb rate, which in turn cools ambient
  temperature/pressure (ISA lapse rate).
- **cruise**: steady mid-power cruise/loiter at altitude. Pass
  `--cruise-repeat N` to extend this leg to emulate a MALE UAV's long
  endurance loiter without altering the shape of the rest of the flight.
- **descent**: throttle eases down; altitude decreases back toward the
  ground.
- **taxi / shutdown**: power reduces to idle then to zero; RPM decays to
  0 and all engine-dependent parameters (oil pressure, fuel flow, EGT,
  etc.) relax toward ambient/zero on their own time constants (so CHT and
  oil temperature cool down slowly after shutdown, exactly as a real
  engine does).

Each row's `operating_condition` column records which phase produced it.
The phase list and every throttle/altitude profile is defined in
`build_default_mission()` and can be edited or replaced entirely if a
different mission shape is needed.

---

## 3. Fault injection (manual, gradual, off by default)

No fault is active unless explicitly requested via `--fault`. When active,
severity follows a deterministic, gradual piecewise-linear curve:

```
0 ───(ramp_s)───> severity_max ───(hold_s)───> [optional recover_s back to 0]
```

`fault_severity` (0.0–1.0) is written to every row, so datasets are fully
labeled for supervised ML use, and the fault's effect on each sensor
scales continuously with that severity — never as a step change.

### Supported faults and their modeled effects

1. **overheating** — CHT, EGT and oil temperature targets are all
   increased in proportion to severity (CHT +45 °C, EGT +65 °C, oil temp
   +22 °C at severity 1.0), still filtered through each parameter's normal
   thermal time constant so the rise is gradual.
2. **low_oil_pressure** — oil pressure target is reduced by up to 75% at
   full severity; the resulting extra friction also nudges oil temperature
   upward, and at high severity (>0.6) vibration increases slightly.
3. **high_vibration** — vibration baseline increases by up to +0.45 g and
   its noise floor widens, simulating a developing mechanical imbalance
   (e.g. bearing wear, prop imbalance).
4. **misfire** — intermittent RPM dropouts and reduced effective load are
   injected stochastically (frequency/magnitude scaled by severity), EGT
   oscillates (a weak/dead cylinder alternately runs cool and puffs
   unburned fuel), and vibration rises moderately.
5. **fuel_flow_anomaly** — fuel flow oscillates around a biased mean
   (chosen once per fault instance as either a lean or rich bias) with
   amplitude scaling with severity; fuel pressure fluctuates in step; EGT
   is coupled to the same lean/rich bias (lean → hotter EGT, rich →
   cooler EGT), consistent with real mixture-related exhaust behavior.

All fault parameters (`--fault-start`, `--fault-ramp`, `--fault-hold`,
`--fault-recover`, `--severity-max`) are configurable from the CLI.

---

## 4. Output format

One CSV row per `--dt` seconds (default 1.0 s), columns:

```
timestamp, rpm, throttle_percent, engine_load_percent, cht_c, egt_c,
oil_pressure_bar, oil_temperature_c, fuel_flow_l_h, fuel_pressure_bar,
vibration_g, voltage_v, ambient_temperature_c, ambient_pressure_hpa,
operating_condition, fault_type, fault_severity
```

- `fault_type` is `"normal"` for every row unless a fault was injected and
  has non-zero severity at that timestamp, in which case it is the fault
  name (`overheating`, `low_oil_pressure`, `high_vibration`, `misfire`,
  `fuel_flow_anomaly`).
- `fault_severity` is `0.0` during normal operation and rises/falls
  smoothly with the injected fault.

---

## 5. Usage

```bash
# Normal-operation dataset (single full flight, 1 row/second)
python aero_engine_simulator.py --output engine_telemetry_normal.csv

# One fault scenario (e.g. gradual overheating starting partway through cruise)
python aero_engine_simulator.py --fault overheating \
    --fault-start 900 --fault-ramp 120 --output engine_telemetry_overheat.csv

# A fault that ramps up, holds, then recovers (e.g. transient high vibration)
python aero_engine_simulator.py --fault high_vibration \
    --fault-start 700 --fault-ramp 60 --fault-hold 300 --fault-recover 120 \
    --output engine_telemetry_vibration_transient.csv

# Generate one normal file + one file per fault type in a single command
python aero_engine_simulator.py --fault all --output-dir telemetry_output

# Longer-endurance mission (extra cruise legs) at 1 Hz
python aero_engine_simulator.py --cruise-repeat 6 --output engine_telemetry_long.csv
```

### Key CLI options

| Flag | Meaning | Default |
|---|---|---|
| `--output` | Output CSV path (single-run mode) | `engine_telemetry.csv` |
| `--output-dir` | Directory for `--fault all` mode | `telemetry_output` |
| `--dt` | Seconds between records | `1.0` |
| `--cruise-repeat` | Repeat the cruise leg N times (long endurance) | `1` |
| `--seed` | Random seed (noise + fault stochastics reproducible) | `42` |
| `--fault` | `none`, one of the 5 fault names, or `all` | `none` |
| `--fault-start` | Sim time (s) fault begins | 40% into mission |
| `--fault-ramp` | Seconds to reach full severity | `90` |
| `--fault-hold` | Seconds held at full severity (omit = until end) | until end |
| `--fault-recover` | Seconds to ramp back to 0 after hold | `0` (no recovery) |
| `--severity-max` | Cap on severity, 0–1 | `1.0` |

---

## 6. Extending / configuring

- All engine constants, coefficients, time constants, and noise levels are
  in the `EngineConfig` dataclass — edit them to match a different real
  engine or to widen/narrow sensor noise.
- The mission profile is built by `build_default_mission()` — add,
  remove, or reshape `Phase` entries (each with its own throttle profile
  and altitude rate) to model a different flight plan.
- New fault types can be added by extending `FAULT_TYPES` and adding a
  corresponding effect block inside `EngineSimulator.step()`.

This generator does not include any ML model, anomaly detector, or health
score — its output CSVs are meant to be fed into a separate model-training
or analysis pipeline of your choosing.
