# Crop Recommendation dataset

Place your real dataset here as `crop_recommendation.csv` with these columns:

```
N, P, K, temperature, humidity, ph, rainfall, label
```

- `N`, `P`, `K`: soil Nitrogen, Phosphorus, Potassium content (numeric)
- `temperature`: °C
- `humidity`: % relative humidity
- `ph`: soil pH (0–14)
- `rainfall`: mm
- `label`: target crop name (string)

This matches the public Kaggle "Crop Recommendation Dataset" schema and
the inputs described in Section III-B of the project paper.

## Current data

`crop_recommendation.csv` in this folder **is the real dataset** (2200
rows, 22 crop classes, 100 samples/class, no missing values), pulled from
a GitHub mirror of the Kaggle "Crop Recommendation Dataset". Validated
model performance on the held-out test split: **99.4% accuracy, 0.994
macro F1**. Minor confusion only between blackgram/cotton and
lentil/maize, consistent with published results on this dataset.
