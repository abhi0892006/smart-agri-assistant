"""
IoT sensor simulator.

The project paper's IoT layer (soil moisture/temperature/pH sensors, an
NPK sensor, an air temperature/humidity sensor, a rain gauge, and a light
sensor, feeding a server over MQTT/HTTP via an ESP32/ESP8266) is
explicitly described as conceptual future work — no real deployment
exists. This module fills that gap with a *simulator*, not a synthetic
dataset: it generates one plausible sensor reading at a time, following
a simple diurnal (day/night) cycle for temperature/humidity/light rather
than pure random noise, so downstream models see physically sensible
input rather than static random numbers.

This is intentionally the one module in the project that is NOT backed
by real data, because the paper itself says no real sensor data exists
yet — see data/../README files in the other four modules for contrast.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
import config


@dataclass
class SensorReading:
    """One simulated snapshot from the full sensor suite."""
    hour_of_day: float          # 0-24
    N: float
    P: float
    K: float
    ph: float
    rainfall_mm_per_day: float
    air_temperature_c: float
    humidity_pct: float
    soil_moisture_pct: float    # for fertilizer_recommendation's scale (0-100ish)
    soil_moisture_raw: float    # for irrigation_prediction's raw sensor scale
    soil_temperature_c: float
    light_lux: float

    def as_dict(self) -> dict:
        return asdict(self)


class IoTSensorSimulator:
    """Generates one simulated sensor reading per call to `read()`.

    A simple internal clock (`hour_of_day`) advances each call so
    repeated reads trace out a plausible day/night cycle rather than
    being independent random samples.
    """

    def __init__(self, seed: int | None = None, hours_per_step: float = 1.0):
        self._rng = random.Random(seed if seed is not None else config.IOT_RANDOM_SEED)
        self.hour_of_day = self._rng.uniform(0, 24)
        self.hours_per_step = hours_per_step
        # Running soil moisture state (two independent scales, see
        # config.IOT_SENSOR_RANGES comment): each depletes over time,
        # "recharged" by rainfall or an external irrigation event.
        pct_lo, pct_hi = config.IOT_SENSOR_RANGES["soil_moisture_pct"]
        raw_lo, raw_hi = config.IOT_SENSOR_RANGES["soil_moisture_raw"]
        self._soil_moisture_pct = self._rng.uniform(pct_lo, pct_hi)
        self._soil_moisture_raw = self._rng.uniform(raw_lo, raw_hi)

    def _diurnal_factor(self) -> float:
        """Returns ~0 at midnight, ~1 at midday, following a smooth cosine cycle."""
        radians = (self.hour_of_day / 24.0) * 2 * math.pi
        return (1 - math.cos(radians)) / 2  # 0 at hour 0, 1 at hour 12, 0 at hour 24

    def irrigate(self, amount: float = 200.0) -> None:
        """Simulate an irrigation event bumping both soil moisture readings up."""
        pct_lo, pct_hi = config.IOT_SENSOR_RANGES["soil_moisture_pct"]
        raw_lo, raw_hi = config.IOT_SENSOR_RANGES["soil_moisture_raw"]
        self._soil_moisture_pct = min(pct_hi, self._soil_moisture_pct + amount * (pct_hi - pct_lo) / (raw_hi - raw_lo))
        self._soil_moisture_raw = min(raw_hi, self._soil_moisture_raw + amount)

    def read(self) -> SensorReading:
        ranges = config.IOT_SENSOR_RANGES
        diurnal = self._diurnal_factor()

        # Temperature and light follow the day/night cycle; humidity is
        # roughly inverse to temperature (drier air when it's hotter).
        temp_lo, temp_hi = ranges["air_temperature_c"]
        air_temp = temp_lo + (temp_hi - temp_lo) * diurnal + self._rng.uniform(-1.5, 1.5)

        hum_lo, hum_hi = ranges["humidity_pct"]
        humidity = hum_hi - (hum_hi - hum_lo) * diurnal + self._rng.uniform(-3, 3)
        humidity = max(hum_lo, min(hum_hi, humidity))

        light_lo, light_hi = ranges["light_lux"]
        light = light_hi * diurnal * self._rng.uniform(0.85, 1.0)

        soil_temp_lo, soil_temp_hi = ranges["soil_temperature_c"]
        # Soil temperature lags and dampens the air temperature swing.
        soil_temp = soil_temp_lo + (soil_temp_hi - soil_temp_lo) * (0.3 + 0.4 * diurnal)

        rain_lo, rain_hi = ranges["rainfall_mm_per_day"]
        # Rain is rare in any given reading, occasionally heavy.
        rainfall = 0.0 if self._rng.random() > 0.15 else self._rng.uniform(rain_lo, rain_hi)
        pct_lo, pct_hi = ranges["soil_moisture_pct"]
        raw_lo, raw_hi = ranges["soil_moisture_raw"]
        if rainfall > 0:
            self._soil_moisture_pct = min(pct_hi, self._soil_moisture_pct + rainfall * (pct_hi - pct_lo) / rain_hi)
            self._soil_moisture_raw = min(raw_hi, self._soil_moisture_raw + rainfall * (raw_hi - raw_lo) / rain_hi)

        # Soil moisture depletes gradually each step (evapotranspiration),
        # unless topped up by rain/irrigation above.
        self._soil_moisture_pct = max(
            pct_lo, self._soil_moisture_pct - self._rng.uniform(1, 4) * self.hours_per_step
        )
        self._soil_moisture_raw = max(
            raw_lo, self._soil_moisture_raw - self._rng.uniform(5, 20) * self.hours_per_step
        )

        n_lo, n_hi = ranges["N"]
        p_lo, p_hi = ranges["P"]
        k_lo, k_hi = ranges["K"]
        ph_lo, ph_hi = ranges["ph"]

        reading = SensorReading(
            hour_of_day=round(self.hour_of_day, 2),
            N=round(self._rng.uniform(n_lo, n_hi), 1),
            P=round(self._rng.uniform(p_lo, p_hi), 1),
            K=round(self._rng.uniform(k_lo, k_hi), 1),
            ph=round(self._rng.uniform(ph_lo, ph_hi), 2),
            rainfall_mm_per_day=round(rainfall, 1),
            air_temperature_c=round(air_temp, 1),
            humidity_pct=round(humidity, 1),
            soil_moisture_pct=round(self._soil_moisture_pct, 1),
            soil_moisture_raw=round(self._soil_moisture_raw, 1),
            soil_temperature_c=round(soil_temp, 1),
            light_lux=round(light, 0),
        )

        self.hour_of_day = (self.hour_of_day + self.hours_per_step) % 24
        return reading


if __name__ == "__main__":
    sim = IoTSensorSimulator(seed=7, hours_per_step=3)
    print("Simulating 8 readings (24 hours, one every 3 hours):\n")
    for _ in range(8):
        r = sim.read()
        print(
            f"hour={r.hour_of_day:5.1f} | air_temp={r.air_temperature_c:5.1f}C | "
            f"humidity={r.humidity_pct:5.1f}% | light={r.light_lux:7.0f}lux | "
            f"soil_moisture_pct={r.soil_moisture_pct:5.1f} | soil_moisture_raw={r.soil_moisture_raw:5.1f} | "
            f"rainfall={r.rainfall_mm_per_day:5.1f}mm"
        )
