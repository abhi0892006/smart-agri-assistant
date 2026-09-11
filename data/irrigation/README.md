# Irrigation Prediction dataset

`irrigation_prediction.csv` in this folder is the real "Irrigation
Dataset" published by the GCEK IoT Community (a joint effort between two
government engineering colleges in Odisha, India, under an NPIU/MHRD
collaborative research scheme). Source:
https://github.com/GCEKIoTCommunity/Irrigation-Dataset (original file is
`Project_datasheet_2019-2020.xlsx`; converted to CSV here for consistency
with the other modules).

Columns:

```
CropType, CropDays, Soil Moisture, Soil Temperature, Temperature,
Humidity, Irrigation(Y/N)
```

- `CropType`: 1 = Paddy, 2 = Ground Nuts (see `config.IRRIGATION_CROP_TYPE_MAP`)
- `CropDays`: days since planting
- `Soil Moisture`, `Soil Temperature`, `Temperature`, `Humidity`: sensor readings
- `Irrigation(Y/N)`: target — 1 = irrigation needed, 0 = not needed

## Known limitations

- **Small**: 150 rows.
- **Imbalanced**: 125 "No" vs 25 "Yes" (~83%/17%). A model that always
  predicts "No" would already score 83% accuracy, so accuracy alone is
  not a meaningful metric here — the training script uses a
  class-weighted loss and reports per-class precision/recall for this
  reason.
- **Only 2 crop types**: Paddy and Ground Nuts. The model will flag a
  warning (via `IrrigationPredictor.predict()`) if asked about any other
  crop type, since it has no training signal for it.

Validated performance: 83% overall accuracy, 80% recall on the minority
"Yes" (irrigate) class, 50% precision on "Yes" (some false positives).
Treat this as a reasonable pipeline demonstration, not a production
irrigation controller — a real deployment needs much more data across
more crop types and geographies.
