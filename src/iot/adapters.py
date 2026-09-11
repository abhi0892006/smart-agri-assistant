"""
Adapters: convert one raw `SensorReading` into the exact input schema
each trained model expects.

The three tabular modules were trained on three separate public datasets
that don't share a naming convention for crop types (crop_recommendation
uses lowercase names like "rice"; fertilizer_recommendation and
irrigation_prediction use "Paddy"). A real deployment would need a single
canonical crop taxonomy mapped across all three — that mapping doesn't
exist yet, so these adapters take the target crop/soil context as
explicit arguments rather than guessing, and this limitation is called
out in src/iot/README.md.
"""
from __future__ import annotations

from src.iot.simulator import SensorReading


def to_crop_recommendation_input(reading: SensorReading) -> dict:
    """No crop/soil-type context needed — crop_recommendation predicts
    the crop itself from raw soil + weather readings."""
    return {
        "N": reading.N,
        "P": reading.P,
        "K": reading.K,
        "temperature": reading.air_temperature_c,
        "humidity": reading.humidity_pct,
        "ph": reading.ph,
        "rainfall": reading.rainfall_mm_per_day,
    }


def to_fertilizer_recommendation_input(reading: SensorReading, soil_type: str, crop_type: str) -> dict:
    """Requires soil_type/crop_type context (e.g. 'Sandy', 'Maize') since
    fertilizer_recommendation was trained with those as categorical inputs,
    not predicted from sensor data alone."""
    return {
        "Temparature": reading.air_temperature_c,
        "Humidity": reading.humidity_pct,
        "Moisture": reading.soil_moisture_pct,
        "Soil Type": soil_type,
        "Crop Type": crop_type,
        "Nitrogen": reading.N,
        "Potassium": reading.K,
        "Phosphorous": reading.P,
    }


def to_irrigation_prediction_input(reading: SensorReading, crop_type_code: int, crop_days: int) -> dict:
    """Requires crop_type_code (1=Paddy, 2=Ground Nuts, see
    config.IRRIGATION_CROP_TYPE_MAP) and crop_days (days since planting),
    since irrigation_prediction was trained with those as inputs."""
    return {
        "CropType": crop_type_code,
        "CropDays": crop_days,
        "Soil Moisture": reading.soil_moisture_raw,
        "Soil Temperature": reading.soil_temperature_c,
        "Temperature": reading.air_temperature_c,
        "Humidity": reading.humidity_pct,
    }
