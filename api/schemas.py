"""
Pydantic request/response schemas for the Smart Agriculture Assistant API.

Field names deliberately match each trained model's native input schema
(including the source datasets' own quirks like "Temparature" and
"Phosphorous") so there's no hidden renaming between what a client sends
and what the model actually saw during training.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CropRecommendationRequest(BaseModel):
    N: float = Field(..., description="Soil Nitrogen content")
    P: float = Field(..., description="Soil Phosphorus content")
    K: float = Field(..., description="Soil Potassium content")
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Relative humidity, percent")
    ph: float = Field(..., description="Soil pH (0-14)")
    rainfall: float = Field(..., description="Rainfall in mm")

    class Config:
        json_schema_extra = {
            "example": {
                "N": 90, "P": 42, "K": 43,
                "temperature": 20.8, "humidity": 82.0,
                "ph": 6.5, "rainfall": 202.9,
            }
        }


class FertilizerRecommendationRequest(BaseModel):
    Temparature: float = Field(..., description="Temperature in Celsius (dataset's own spelling)")
    Humidity: float = Field(..., description="Relative humidity, percent")
    Moisture: float = Field(..., description="Soil moisture, ~0-100 scale (see src/iot/README.md for units caveat)")
    Soil_Type: str = Field(..., alias="Soil Type", description="One of: Sandy, Loamy, Black, Red, Clayey")
    Crop_Type: str = Field(..., alias="Crop Type", description="See FertilizerRecommender.valid_crop_types()")
    Nitrogen: float
    Potassium: float
    Phosphorous: float

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Temparature": 26, "Humidity": 52, "Moisture": 38,
                "Soil Type": "Sandy", "Crop Type": "Maize",
                "Nitrogen": 37, "Potassium": 0, "Phosphorous": 0,
            }
        }


class IrrigationPredictionRequest(BaseModel):
    CropType: int = Field(..., description="1=Paddy, 2=Ground Nuts")
    CropDays: int = Field(..., description="Days since planting")
    Soil_Moisture: float = Field(..., alias="Soil Moisture", description="Raw sensor reading, ~100-900 scale")
    Soil_Temperature: float = Field(..., alias="Soil Temperature", description="Celsius")
    Temperature: float = Field(..., description="Air temperature, Celsius")
    Humidity: float = Field(..., description="Relative humidity, percent")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "CropType": 1, "CropDays": 3, "Soil Moisture": 230,
                "Soil Temperature": 25, "Temperature": 30, "Humidity": 60,
            }
        }


class FieldReportRequest(BaseModel):
    """Optional scenario context for the combined /iot/field-report endpoint.
    See src/iot/adapters.py for why crop/soil context can't be auto-derived."""
    soil_type: str = "Sandy"
    crop_type: str = "Maize"
    irrigation_crop_code: int = 1
    crop_days: int = 15
