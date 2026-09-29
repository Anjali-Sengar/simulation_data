#!/usr/bin/env python3
"""
aero_engine_simulator.py
=========================

Continuous synthetic telemetry generator for a SINGLE virtual aero piston
engine intended for a MALE UAV health-monitoring project.

Engine modeled
--------------
4-cylinder, 4-stroke, horizontally-opposed, air-cooled, gasoline-powered
direct-drive aero piston engine (parameter set is loosely modeled after the
Lycoming/Continental O-235/O-320 class of ~115-160 hp direct-drive
horizontally-opposed engines commonly referenced for this power class of
UAV/light-aircraft powerplant -- see README.md "Assumptions" section for the
exact numeric basis of every parameter).

Design principles (per project requirements)
---------------------------------------------
1. ONE continuous engine is modeled as a single running state machine --
   not independent samples and not multiple engines.
2. Every parameter at time t is derived from its value at time t-dt plus a
   physically-motivated "target" value computed from current operating
   conditions (throttle, RPM, load, altitude, ambient conditions, and any
   active fault). Values move toward their targets using first-order
   (exponential) smoothing with a parameter-specific time constant (tau),
   so fast gas-temperature dynamics (EGT), medium metal-temperature
   dynamics (CHT), and slow oil-thermal-mass dynamics (oil temperature)
   all behave with realistically different responsiveness.
3. Small Gaussian sensor noise is added on top of the smoothed physical
   value to emulate real sensor behavior.
4. Faults are OFF by default and must be explicitly requested. When
   active, a fault's severity ramps up smoothly (0 -> severity_max) over a
   configurable ramp period, optionally holds, and optionally ramps back
   down -- faults never appear as instantaneous step changes.

This file is ONLY a data simulator. It does not implement or call any
machine-learning model, anomaly detector, health-score calculator, or
dashboard. Its only job is to produce labeled CSV telemetry suitable for
later use as input to a separate ML pipeline.
"""

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

# =============================================================================
# 1. ENGINE / ENVIRONMENT / NOISE CONFIGURATION
#    (Every constant below is a documented assumption -- see README.md)
# =============================================================================


@dataclass
class EngineConfig:
    # --- RPM ---
    idle_rpm: float = 650.0          # RPM at closed throttle, engine running
    max_rpm: float = 2700.0          # Full-throttle / redline RPM
    crank_rpm: float = 300.0         # Peak RPM seen while starter is cranking

    # --- Steady-state relationship coefficients ---
    # engine_load_percent = clip(100 * (0.30*rpm_frac + 0.70*throttle_frac) *
    #                             density_ratio**0.6, 0, 100)
    load_rpm_weight: float = 0.30
    load_throttle_weight: float = 0.70
    load_density_exponent: float = 0.6

    # CHT (deg C) = ambient + cht_base + cht_load_coeff * load_percent
    cht_base: float = 99.0
    cht_load_coeff: float = 0.86
    cht_max_c: float = 235.0         # absolute physical ceiling (redline ~232C)

    # EGT (deg C) = ambient + egt_base + egt_load_coeff * load_percent
    egt_base: float = 314.0
    egt_load_coeff: float = 5.16
    egt_max_c: float = 900.0

    # Oil temperature (deg C) = ambient + oil_base + oil_load_coeff * load_percent
    oil_temp_base: float = 45.0
    oil_temp_load_coeff: float = 0.40
    oil_temp_max_c: float = 125.0

    # Oil pressure (bar) = oil_p_floor + oil_p_rpm_gain*rpm_frac
    #                       - oil_p_temp_coeff*(oil_temp - oil_p_temp_ref)
    oil_p_floor: float = 1.0
    oil_p_rpm_gain: float = 5.0
    oil_p_temp_coeff: float = 0.01
    oil_p_temp_ref: float = 70.0
    oil_p_max_bar: float = 7.0

    # Fuel flow (L/h) = fuel_idle_offset + fuel_load_coeff * load_percent (if running)
    fuel_idle_offset: float = 5.0
    fuel_load_coeff: float = 0.45

    # Fuel pressure (bar) = fuel_p_base + fuel_p_rpm_gain * rpm_frac
    fuel_p_base: float = 0.30
    fuel_p_rpm_gain: float = 0.15

    # Vibration (g) baseline = vib_base + vib_rpm_gain*rpm_frac + vib_load_gain*load_frac
    vib_base: float = 0.05
    vib_rpm_gain: float = 0.06
    vib_load_gain: float = 0.02
    vib_off_g: float = 0.01          # residual vibration, engine not running (wind/handling)

    # Electrical
    voltage_battery_resting: float = 12.6   # engine off, no load
    voltage_cranking: float = 9.5           # starter motor drawing heavy current
    voltage_running_base: float = 13.8      # alternator regulated, idle
    voltage_running_rpm_gain: float = 0.4   # extra volts approaching max RPM
    voltage_max: float = 14.4

    # --- Time constants (seconds) -- how fast each smoothed value moves
    # toward its target. Larger tau = slower/more thermally massive.
    tau_rpm: float = 4.0
    tau_throttle: float = 2.0
    tau_load: float = 3.0
    tau_cht: float = 120.0
    tau_egt: float = 20.0
    tau_oil_pressure: float = 5.0
    tau_oil_temp: float = 240.0
    tau_fuel_flow: float = 3.0
    tau_fuel_pressure: float = 2.0
    tau_vibration: float = 5.0
    tau_voltage: float = 3.0
    tau_ambient: float = 60.0

    # --- Sensor noise (Gaussian std-dev, applied every step) ---
    noise_rpm: float = 3.0
    noise_throttle: float = 0.3
    noise_load: float = 0.4
    noise_cht: float = 0.4
    noise_egt: float = 2.0
    noise_oil_pressure: float = 0.03
    noise_oil_temp: float = 0.15
    noise_fuel_flow: float = 0.2
    noise_fuel_pressure: float = 0.01
    noise_vibration: float = 0.006
    noise_voltage: float = 0.05
    noise_ambient_temp: float = 0.05
    noise_ambient_pressure: float = 0.05

    # --- Atmosphere / altitude ---
    sea_level_temp_c: float = 15.0
    sea_level_pressure_hpa: float = 1013.25
    lapse_rate_c_per_m: float = 0.0065
    max_altitude_m: float = 4500.0    # ~ MALE UAV representative operating ceiling
    climb_rate_m_s: float = 5.0       # ~1000 ft/min
    descent_rate_m_s: float = 4.0     # ~800 ft/min
    weather_drift_temp_sigma: float = 0.02   # slow random-walk drift per second
    weather_drift_press_sigma: float = 0.03


# =============================================================================
# 2. MISSION PROFILE (operating-condition schedule)
# =============================================================================


@dataclass
class Phase:
    name: str                      # value written to "operating_condition"
    duration_s: float
    kind: str                      # "start" | "run" | "shutdown"
    throttle_profile: Optional[Callable[[float], float]] = None  # f(frac 0..1) -> %
    altitude_rate: float = 0.0      # m/s applied while in this phase (run phases)


def _ramp(f: float, f0: float, f1: float, v0: float, v1: float) -> float:
    """Linearly interpolate v0->v1 as f goes from f0->f1, clamped outside."""
    if f1 <= f0:
        return v1
    frac = min(max((f - f0) / (f1 - f0), 0.0), 1.0)
    return v0 + (v1 - v0) * frac


def build_default_mission(cfg: EngineConfig, cruise_repeat: int = 1) -> List[Phase]:
    """
    Builds the default flight profile:
    start -> idle -> taxi -> takeoff -> climb -> cruise (xN) -> descent ->
    taxi -> shutdown

    `cruise_repeat` lets the cruise leg be repeated to emulate the long
    endurance loiter typical of a MALE UAV mission without changing the
    shape of every other phase.
    """
    phases: List[Phase] = [
        Phase("start", 20.0, "start"),
        Phase("idle", 60.0, "run", throttle_profile=lambda f: 8.0),
        Phase("taxi", 90.0, "run", throttle_profile=lambda f: 18.0),
        Phase(
            "takeoff", 60.0, "run",
            throttle_profile=lambda f: _ramp(f, 0.0, 0.15, 20.0, 100.0),
            altitude_rate=0.0,
        ),
        Phase(
            "climb", 300.0, "run",
            throttle_profile=lambda f: _ramp(f, 0.0, 0.10, 100.0, 85.0),
            altitude_rate=cfg.climb_rate_m_s,
        ),
    ]
    for _ in range(max(1, cruise_repeat)):
        phases.append(
            Phase("cruise", 600.0, "run", throttle_profile=lambda f: 65.0,
                  altitude_rate=0.0)
        )
    phases.extend([
        Phase(
            "descent", 300.0, "run",
            throttle_profile=lambda f: _ramp(f, 0.0, 0.20, 65.0, 25.0),
            altitude_rate=-cfg.descent_rate_m_s,
        ),
        Phase("taxi", 60.0, "run", throttle_profile=lambda f: 18.0),
        Phase("shutdown", 20.0, "shutdown"),
    ])
    return phases


# =============================================================================
# 3. FAULT INJECTION
# =============================================================================

FAULT_TYPES = [
    "overheating",
    "low_oil_pressure",
    "high_vibration",
    "misfire",
    "fuel_flow_anomaly",
]


@dataclass
class FaultConfig:
    fault_type: Optional[str] = None   # one of FAULT_TYPES, or None for normal
    start_s: float = 0.0                # simulation time the fault begins developing
    ramp_s: float = 60.0                # time to go from 0 -> severity_max
    hold_s: Optional[float] = None      # time held at severity_max; None = until end
    recover_s: float = 0.0              # time to ramp back down to 0 (0 = no recovery)
    severity_max: float = 1.0           # cap in [0, 1]
    rng_seed: int = 42                  # controls fault-specific stochastic effects


def compute_fault_severity(t: float, fc: FaultConfig) -> float:
    """Piecewise-linear severity(t) in [0, severity_max]. Fully deterministic
    and gradual: no fault ever appears as a step change."""
    if fc.fault_type is None or t < fc.start_s:
        return 0.0
    dt1 = t - fc.start_s
    if dt1 < fc.ramp_s:
        return fc.severity_max * (dt1 / fc.ramp_s if fc.ramp_s > 0 else 1.0)
    dt2 = dt1 - fc.ramp_s
    if fc.hold_s is None or dt2 < fc.hold_s:
        return fc.severity_max
    dt3 = dt2 - fc.hold_s
    if fc.recover_s > 0 and dt3 < fc.recover_s:
        return fc.severity_max * (1.0 - dt3 / fc.recover_s)
    return 0.0 if fc.recover_s > 0 else fc.severity_max


# =============================================================================
# 4. ATMOSPHERE HELPERS (ISA model, used so ambient temp/pressure move
#    slowly and consistently with simulated altitude)
# =============================================================================


def isa_temperature_c(altitude_m: float, cfg: EngineConfig) -> float:
    return cfg.sea_level_temp_c - cfg.lapse_rate_c_per_m * altitude_m


def isa_pressure_hpa(altitude_m: float, cfg: EngineConfig) -> float:
    return cfg.sea_level_pressure_hpa * (
        1.0 - 2.25577e-5 * altitude_m
    ) ** 5.25588


# =============================================================================
# 5. SIMULATOR
# =============================================================================


def approach(current: float, target: float, tau: float, dt: float) -> float:
    """First-order exponential approach of `current` toward `target`."""
    if tau <= 0:
        return target
    alpha = 1.0 - math.exp(-dt / tau)
    return current + (target - current) * alpha


class EngineSimulator:
    """
    Stateful, continuous simulator for ONE aero piston engine. Call step(dt)
    repeatedly; every call mutates internal state based on the previous
    state, the current operating condition, and any active fault.
    """

    def __init__(self, cfg: EngineConfig, mission: List[Phase],
                 fault: FaultConfig, rng: random.Random):
        self.cfg = cfg
        self.mission = mission
        self.fault = fault
        self.rng = rng

        # Precompute cumulative phase start times
        self._phase_starts = []
        acc = 0.0
        for p in mission:
            self._phase_starts.append(acc)
            acc += p.duration_s
        self.total_duration = acc

        # --- initial state: engine cold and off, on the ground ---
        self.t = 0.0
        self.rpm = 0.0
        self.throttle = 0.0
        self.load = 0.0
        self.altitude_m = 0.0
        self.ambient_temp = cfg.sea_level_temp_c
        self.ambient_pressure = cfg.sea_level_pressure_hpa
        self.cht = self.ambient_temp
        self.egt = self.ambient_temp
        self.oil_pressure = 0.0
        self.oil_temp = self.ambient_temp
        self.fuel_flow = 0.0
        self.fuel_pressure = 0.0
        self.vibration = cfg.vib_off_g
        self.voltage = cfg.voltage_battery_resting

        # Fault-specific persistent random state (kept stable across the
        # whole fault event rather than re-randomized every tick)
        fault_rng = random.Random(fault.rng_seed)
        self._fuel_anomaly_direction = fault_rng.choice([-1.0, 1.0])
        self._fuel_anomaly_period = fault_rng.uniform(25.0, 45.0)
        self._misfire_phase_offset = fault_rng.uniform(0.0, 2 * math.pi)

    # -- phase lookup -------------------------------------------------
    def _current_phase(self) -> Tuple[Phase, float]:
        """Returns (phase, fraction_through_phase in [0,1])."""
        for i, p in enumerate(self.mission):
            start = self._phase_starts[i]
            end = start + p.duration_s
            if self.t < end or i == len(self.mission) - 1:
                frac = 0.0 if p.duration_s <= 0 else min(
                    max((self.t - start) / p.duration_s, 0.0), 1.0)
                return p, frac
        return self.mission[-1], 1.0

    # -- one simulation tick ------------------------------------------
    def step(self, dt: float) -> Dict[str, object]:
        cfg = self.cfg
        phase, frac = self._current_phase()
        severity = compute_fault_severity(self.t, self.fault)
        fault_type = self.fault.fault_type if severity > 0.0 else None

        # ---------------------------------------------------------------
        # Throttle & RPM target (start/shutdown phases use direct RPM
        # ramps to represent cranking / spool-down; run phases derive RPM
        # from throttle so "increasing throttle raises RPM" causally)
        # ---------------------------------------------------------------
        if phase.kind == "start":
            crank_frac = 0.35
            if frac < crank_frac:
                throttle_target = 3.0
                rpm_target = _ramp(frac, 0.0, crank_frac, 0.0, cfg.crank_rpm)
                is_cranking = True
            else:
                throttle_target = 8.0
                rpm_target = _ramp(frac, crank_frac, 1.0, cfg.crank_rpm, cfg.idle_rpm)
                is_cranking = False
        elif phase.kind == "shutdown":
            throttle_target = _ramp(frac, 0.0, 0.25, self.throttle, 0.0)
            rpm_target = _ramp(frac, 0.0, 1.0, self.rpm, 0.0)
            is_cranking = False
        else:  # "run"
            throttle_target = phase.throttle_profile(frac)
            # Misfire drops effective throttle response / power delivery
            if fault_type == "misfire":
                dropout = 1.0 - 0.35 * severity * (
                    0.5 + 0.5 * math.sin(self.t / 3.5 + self._misfire_phase_offset)
                )
                throttle_target *= max(0.4, dropout)
            rpm_target = cfg.idle_rpm + (cfg.max_rpm - cfg.idle_rpm) * (
                max(throttle_target, 0.0) / 100.0
            )
            is_cranking = False

        self.throttle = approach(self.throttle, throttle_target, cfg.tau_throttle, dt)
        self.throttle += self.rng.gauss(0.0, cfg.noise_throttle)
        self.throttle = min(max(self.throttle, 0.0), 100.0)

        self.rpm = approach(self.rpm, rpm_target, cfg.tau_rpm, dt)
        # Misfire: intermittent RPM dropouts (single-cylinder-loss stutter)
        if fault_type == "misfire" and self.rng.random() < 0.10 * severity:
            self.rpm -= self.rpm * 0.06 * severity
        self.rpm += self.rng.gauss(0.0, cfg.noise_rpm)
        self.rpm = max(self.rpm, 0.0)

        is_running = self.rpm > 50.0
        rpm_frac = min(self.rpm / cfg.max_rpm, 1.0)
        throttle_frac = self.throttle / 100.0

        # ---------------------------------------------------------------
        # Altitude & ambient conditions (slow-changing atmosphere)
        # ---------------------------------------------------------------
        if phase.kind == "run":
            self.altitude_m += phase.altitude_rate * dt
            self.altitude_m = min(max(self.altitude_m, 0.0), cfg.max_altitude_m)

        ambient_temp_target = isa_temperature_c(self.altitude_m, cfg)
        ambient_pressure_target = isa_pressure_hpa(self.altitude_m, cfg)
        # slow random-walk "weather" drift on top of the altitude signal
        ambient_temp_target += self.rng.gauss(0.0, cfg.weather_drift_temp_sigma) * dt
        ambient_pressure_target += self.rng.gauss(0.0, cfg.weather_drift_press_sigma) * dt

        self.ambient_temp = approach(self.ambient_temp, ambient_temp_target,
                                      cfg.tau_ambient, dt)
        self.ambient_temp += self.rng.gauss(0.0, cfg.noise_ambient_temp)
        self.ambient_pressure = approach(self.ambient_pressure, ambient_pressure_target,
                                          cfg.tau_ambient, dt)
        self.ambient_pressure += self.rng.gauss(0.0, cfg.noise_ambient_pressure)

        density_ratio = max(self.ambient_pressure / cfg.sea_level_pressure_hpa, 0.3)

        # ---------------------------------------------------------------
        # Engine load (derived from throttle + rpm + air density)
        # ---------------------------------------------------------------
        if is_running:
            load_target = 100.0 * (
                cfg.load_rpm_weight * rpm_frac + cfg.load_throttle_weight * throttle_frac
            ) * (density_ratio ** cfg.load_density_exponent)
            if fault_type == "misfire":
                load_target *= (1.0 - 0.25 * severity)
        else:
            load_target = 0.0
        self.load = approach(self.load, load_target, cfg.tau_load, dt)
        self.load += self.rng.gauss(0.0, cfg.noise_load)
        self.load = min(max(self.load, 0.0), 100.0)

        # ---------------------------------------------------------------
        # Thermal parameters: CHT, EGT, oil temperature
        # ---------------------------------------------------------------
        if is_running:
            cht_target = self.ambient_temp + cfg.cht_base + cfg.cht_load_coeff * self.load
            egt_target = self.ambient_temp + cfg.egt_base + cfg.egt_load_coeff * self.load
            oil_temp_target = self.ambient_temp + cfg.oil_temp_base + \
                cfg.oil_temp_load_coeff * self.load
        else:
            cht_target = self.ambient_temp
            egt_target = self.ambient_temp
            oil_temp_target = self.ambient_temp

        if fault_type == "overheating":
            cht_target += 45.0 * severity
            egt_target += 65.0 * severity
            oil_temp_target += 22.0 * severity
        if fault_type == "misfire":
            # A dead/weak cylinder runs cooler while unburned fuel can spike
            # EGT in the exhaust intermittently -> net oscillation.
            egt_target += 40.0 * severity * math.sin(
                self.t / 6.0 + self._misfire_phase_offset
            ) - 25.0 * severity
        if fault_type == "fuel_flow_anomaly":
            # lean (fuel-flow deficit) -> hotter EGT; rich -> cooler EGT
            egt_target -= 55.0 * severity * self._fuel_anomaly_direction

        self.cht = approach(self.cht, cht_target, cfg.tau_cht, dt)
        self.cht += self.rng.gauss(0.0, cfg.noise_cht)
        self.cht = min(self.cht, cfg.cht_max_c)

        self.egt = approach(self.egt, egt_target, cfg.tau_egt, dt)
        self.egt += self.rng.gauss(0.0, cfg.noise_egt)
        self.egt = min(max(self.egt, self.ambient_temp), cfg.egt_max_c)

        self.oil_temp = approach(self.oil_temp, oil_temp_target, cfg.tau_oil_temp, dt)
        self.oil_temp += self.rng.gauss(0.0, cfg.noise_oil_temp)
        self.oil_temp = min(max(self.oil_temp, self.ambient_temp), cfg.oil_temp_max_c)

        # ---------------------------------------------------------------
        # Oil pressure (depends on RPM and oil temperature/viscosity)
        # ---------------------------------------------------------------
        if is_running:
            oil_pressure_target = (
                cfg.oil_p_floor + cfg.oil_p_rpm_gain * rpm_frac
                - cfg.oil_p_temp_coeff * (self.oil_temp - cfg.oil_p_temp_ref)
            )
        else:
            oil_pressure_target = 0.0

        if fault_type == "low_oil_pressure":
            oil_pressure_target *= (1.0 - 0.75 * severity)
            oil_temp_target_bonus = 15.0 * severity  # extra friction heat
            self.oil_temp += oil_temp_target_bonus * dt / cfg.tau_oil_temp

        self.oil_pressure = approach(self.oil_pressure, oil_pressure_target,
                                      cfg.tau_oil_pressure, dt)
        self.oil_pressure += self.rng.gauss(0.0, cfg.noise_oil_pressure)
        self.oil_pressure = min(max(self.oil_pressure, 0.0), cfg.oil_p_max_bar)

        # ---------------------------------------------------------------
        # Fuel flow & fuel pressure
        # ---------------------------------------------------------------
        if is_running:
            fuel_flow_target = cfg.fuel_idle_offset + cfg.fuel_load_coeff * self.load
            fuel_pressure_target = cfg.fuel_p_base + cfg.fuel_p_rpm_gain * rpm_frac
        else:
            fuel_flow_target = 0.0
            fuel_pressure_target = 0.0

        if fault_type == "fuel_flow_anomaly" and is_running:
            osc = math.sin(2 * math.pi * self.t / self._fuel_anomaly_period)
            fuel_flow_target *= (1.0 + self._fuel_anomaly_direction * 0.45 * severity * osc
                                  - 0.15 * severity)
            fuel_pressure_target += self._fuel_anomaly_direction * 0.5 * severity * osc

        self.fuel_flow = approach(self.fuel_flow, fuel_flow_target, cfg.tau_fuel_flow, dt)
        self.fuel_flow += self.rng.gauss(0.0, cfg.noise_fuel_flow)
        self.fuel_flow = max(self.fuel_flow, 0.0)

        self.fuel_pressure = approach(self.fuel_pressure, fuel_pressure_target,
                                       cfg.tau_fuel_pressure, dt)
        self.fuel_pressure += self.rng.gauss(0.0, cfg.noise_fuel_pressure)
        self.fuel_pressure = max(self.fuel_pressure, 0.0)

        # ---------------------------------------------------------------
        # Vibration
        # ---------------------------------------------------------------
        if is_running:
            vibration_target = (
                cfg.vib_base + cfg.vib_rpm_gain * rpm_frac
                + cfg.vib_load_gain * (self.load / 100.0)
            )
        else:
            vibration_target = cfg.vib_off_g

        if fault_type == "high_vibration":
            vibration_target += 0.45 * severity
        if fault_type == "misfire":
            vibration_target += 0.18 * severity
        if fault_type == "low_oil_pressure" and severity > 0.6:
            vibration_target += 0.10 * (severity - 0.6) / 0.4

        extra_noise = cfg.noise_vibration
        if fault_type == "high_vibration":
            extra_noise += 0.05 * severity  # rougher / higher-amplitude noise floor

        self.vibration = approach(self.vibration, vibration_target, cfg.tau_vibration, dt)
        self.vibration += self.rng.gauss(0.0, extra_noise)
        self.vibration = max(self.vibration, 0.0)

        # ---------------------------------------------------------------
        # Electrical system
        # ---------------------------------------------------------------
        if is_cranking:
            voltage_target = cfg.voltage_cranking
        elif is_running:
            voltage_target = min(
                cfg.voltage_running_base + cfg.voltage_running_rpm_gain * rpm_frac,
                cfg.voltage_max,
            )
        else:
            voltage_target = cfg.voltage_battery_resting

        self.voltage = approach(self.voltage, voltage_target, cfg.tau_voltage, dt)
        self.voltage += self.rng.gauss(0.0, cfg.noise_voltage)

        # ---------------------------------------------------------------
        # Advance time and emit record
        # ---------------------------------------------------------------
        row = {
            "rpm": round(self.rpm, 1),
            "throttle_percent": round(self.throttle, 2),
            "engine_load_percent": round(self.load, 2),
            "cht_c": round(self.cht, 2),
            "egt_c": round(self.egt, 2),
            "oil_pressure_bar": round(self.oil_pressure, 3),
            "oil_temperature_c": round(self.oil_temp, 2),
            "fuel_flow_l_h": round(self.fuel_flow, 2),
            "fuel_pressure_bar": round(self.fuel_pressure, 3),
            "vibration_g": round(self.vibration, 4),
            "voltage_v": round(self.voltage, 2),
            "ambient_temperature_c": round(self.ambient_temp, 2),
            "ambient_pressure_hpa": round(self.ambient_pressure, 2),
            "operating_condition": phase.name,
            "fault_type": fault_type if fault_type else "normal",
            "fault_severity": round(severity, 3),
        }
        self.t += dt
        return row


# =============================================================================
# 6. DATASET GENERATION / CSV OUTPUT
# =============================================================================

CSV_COLUMNS = [
    "timestamp",
    "rpm",
    "throttle_percent",
    "engine_load_percent",
    "cht_c",
    "egt_c",
    "oil_pressure_bar",
    "oil_temperature_c",
    "fuel_flow_l_h",
    "fuel_pressure_bar",
    "vibration_g",
    "voltage_v",
    "ambient_temperature_c",
    "ambient_pressure_hpa",
    "operating_condition",
    "fault_type",
    "fault_severity",
]


def generate_csv(
    output_path: str,
    cfg: EngineConfig,
    fault: FaultConfig,
    dt: float = 1.0,
    cruise_repeat: int = 1,
    seed: int = 42,
    start_time: Optional[datetime] = None,
) -> int:
    """Runs the simulator across the whole mission and writes one CSV row
    every `dt` seconds. Returns the number of rows written."""
    rng = random.Random(seed)
    mission = build_default_mission(cfg, cruise_repeat=cruise_repeat)
    sim = EngineSimulator(cfg, mission, fault, rng)

    if start_time is None:
        start_time = datetime.now(timezone.utc)

    n_steps = int(round(sim.total_duration / dt))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        current_time = start_time
        for _ in range(n_steps):
            row = sim.step(dt)
            row["timestamp"] = current_time.isoformat()
            writer.writerow(row)
            current_time += timedelta(seconds=dt)
    return n_steps


# =============================================================================
# 7. CLI
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="Continuous synthetic telemetry generator for a single "
                    "4-cylinder horizontally-opposed air-cooled aero piston "
                    "engine (MALE UAV health-monitoring use case)."
    )
    p.add_argument("--output", default="engine_telemetry.csv",
                    help="Output CSV path (ignored if --fault all is used; "
                         "see --output-dir instead). Default: engine_telemetry.csv")
    p.add_argument("--output-dir", default="telemetry_output",
                    help="Directory used when --fault all generates one file "
                         "per scenario. Default: telemetry_output")
    p.add_argument("--dt", type=float, default=1.0,
                    help="Seconds between telemetry records. Default: 1.0")
    p.add_argument("--cruise-repeat", type=int, default=1,
                    help="Number of consecutive cruise legs (extends the "
                         "long-endurance loiter portion of the mission). "
                         "Default: 1")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    p.add_argument("--fault", choices=["none"] + FAULT_TYPES + ["all"], default="none",
                    help="Fault scenario to inject. 'all' writes one labeled "
                         "CSV per fault type (plus a normal-operation CSV) "
                         "into --output-dir. Default: none")
    p.add_argument("--fault-start", type=float, default=None,
                    help="Simulation time (s) the fault begins developing. "
                         "Default: 40%% into the mission.")
    p.add_argument("--fault-ramp", type=float, default=90.0,
                    help="Seconds for the fault to ramp from 0 to full "
                         "severity. Default: 90")
    p.add_argument("--fault-hold", type=float, default=None,
                    help="Seconds held at full severity after the ramp. "
                         "Default: held until the end of the mission.")
    p.add_argument("--fault-recover", type=float, default=0.0,
                    help="Seconds to ramp the fault back down to 0 after "
                         "the hold period. 0 = no recovery (default).")
    p.add_argument("--severity-max", type=float, default=1.0,
                    help="Cap on fault severity in [0,1]. Default: 1.0")
    return p.parse_args()


def run_single(args, fault_type: Optional[str], output_path: str):
    cfg = EngineConfig()
    mission_len = sum(p.duration_s for p in build_default_mission(
        cfg, cruise_repeat=args.cruise_repeat))
    fault_start = args.fault_start if args.fault_start is not None else 0.4 * mission_len

    fault = FaultConfig(
        fault_type=fault_type,
        start_s=fault_start,
        ramp_s=args.fault_ramp,
        hold_s=args.fault_hold,
        recover_s=args.fault_recover,
        severity_max=args.severity_max,
        rng_seed=args.seed,
    )
    n = generate_csv(
        output_path, cfg, fault,
        dt=args.dt, cruise_repeat=args.cruise_repeat, seed=args.seed,
    )
    label = fault_type if fault_type else "normal"
    print(f"[{label:>18}] wrote {n:5d} rows -> {output_path}")


def main():
    args = parse_args()
    if args.fault == "all":
        os.makedirs(args.output_dir, exist_ok=True)
        run_single(args, None, os.path.join(args.output_dir, "telemetry_normal.csv"))
        for ft in FAULT_TYPES:
            out = os.path.join(args.output_dir, f"telemetry_{ft}.csv")
            run_single(args, ft, out)
    else:
        fault_type = None if args.fault == "none" else args.fault
        run_single(args, fault_type, args.output)


if __name__ == "__main__":
    main()