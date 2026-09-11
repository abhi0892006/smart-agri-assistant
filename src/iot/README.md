# IoT Simulator

**This is the one module in the project that is deliberately synthetic —
not backed by real sensor data.** The paper's IoT layer (soil
moisture/temperature/pH sensors, an NPK sensor, an air temperature/
humidity sensor, a rain gauge, and a light sensor, feeding a server over
MQTT/HTTP via an ESP32/ESP8266) is explicitly described as conceptual
future work; no real deployment exists. Building a real IoT pipeline was
never in scope here — this module exists so the four trained models have
something realistic to consume in the meantime.

## What it does

`src/iot/simulator.py` — `IoTSensorSimulator` generates one plausible
sensor reading per call to `.read()`. It's not random noise: temperature,
humidity, and light follow a diurnal (day/night) cosine cycle, soil
moisture depletes gradually and gets replenished by simulated rain or an
explicit `.irrigate()` call, matching how a real field would behave.

`src/iot/adapters.py` — converts one raw `SensorReading` into the exact
input schema each trained model expects.

`src/iot/demo.py` — ties it all together: simulates one reading, runs it
through crop recommendation, fertilizer recommendation, and irrigation
prediction, and prints a combined field report.

```bash
python -m src.iot.simulator   # see 8 readings tracing a 24h cycle
python -m src.iot.demo        # full pipeline: sensor -> 3 models -> report
```

## A real limitation this surfaced, not glossed over

While wiring this up, the three tabular models' expected inputs turned
out to be inconsistent in two ways that a real deployment would need to
resolve:

1. **Soil moisture units differ between datasets.** `fertilizer_prediction.csv`'s
   `Moisture` column is on a ~25-65 scale (likely a percentage);
   `irrigation_prediction.csv`'s `Soil Moisture` is a raw sensor/ADC
   reading in the ~100-900 range. These are not the same unit, and there's
   no documented conversion between them in either source dataset — so
   the simulator generates `soil_moisture_pct` and `soil_moisture_raw`
   as two independent values rather than inventing an unfounded formula
   to convert one into the other.
2. **Crop naming isn't consistent across datasets.** crop_recommendation
   was trained on lowercase names like `"rice"`; fertilizer_recommendation
   and irrigation_prediction use `"Paddy"`. There's no shared crop
   taxonomy across all three, so `adapters.py` takes crop/soil-type
   context as explicit arguments for fertilizer and irrigation, rather
   than trying to auto-derive it from crop_recommendation's output.

A production system would need a canonical sensor-unit and crop-taxonomy
standard defined once and enforced across every model's training data —
worth flagging for anyone taking this further.
