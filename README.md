# Smart Agriculture Assistant

Implementation of the multimodal deep-learning framework for agricultural
decision support described in the accompanying paper (VIT-AP): multi-crop
plant disease detection, crop recommendation, fertilizer recommendation,
and irrigation prediction, extended with an IoT-based data-acquisition
layer.

## Status

| Module | Status |
|---|---|
| Crop Recommendation | ✅ Implemented + validated on real data (99.4% test accuracy) |
| Fertilizer Recommendation | ✅ Implemented + validated on real data (95% test accuracy; small dataset, see caveat below) |
| Irrigation Prediction | ✅ Implemented + validated on real data (83% accuracy, 80% recall on minority class; see caveat below) |
| Disease Detection (CNN) | ✅ Implemented + validated on real data (93.9% test accuracy; small 8-class subset, see caveat below) |
| IoT simulator layer | ✅ Implemented (deliberately synthetic — see src/iot/README.md) |
| API serving layer | ✅ Implemented — FastAPI, all 4 models + IoT simulator |

## Setup

```bash
pip install -r requirements.txt
```

## Crop Recommendation module

1. Drop your dataset at `data/crop_recommendation/crop_recommendation.csv`
   (see `data/crop_recommendation/README.md` for the expected schema).
2. Train:
   ```bash
   python -m src.crop_recommendation.train
   ```
   This prints accuracy/precision/recall/F1/confusion-matrix on a held-out
   test split and saves the model + scaler + label encoder to `models/`.
3. Run inference:
   ```bash
   python -m src.crop_recommendation.predict
   ```
   Or import `CropRecommender` from `src.crop_recommendation.predict` in
   your own code:
   ```python
   from src.crop_recommendation.predict import CropRecommender

   rec = CropRecommender()
   result = rec.recommend({
       "N": 90, "P": 42, "K": 43,
       "temperature": 20.8, "humidity": 82.0,
       "ph": 6.5, "rainfall": 202.9,
   })
   print(result)
   ```

## Fertilizer Recommendation module

1. Dataset is already included at `data/fertilizer/fertilizer_prediction.csv`
   (see that folder's README for the size/class-balance caveat).
2. Train:
   ```bash
   python -m src.fertilizer_recommendation.train
   ```
3. Predict:
   ```bash
   python -m src.fertilizer_recommendation.predict
   ```
   Or import directly:
   ```python
   from src.fertilizer_recommendation.predict import FertilizerRecommender

   rec = FertilizerRecommender()
   result = rec.recommend({
       "Temparature": 26, "Humidity": 52, "Moisture": 38,
       "Soil Type": "Sandy", "Crop Type": "Maize",
       "Nitrogen": 37, "Potassium": 0, "Phosphorous": 0,
   })
   print(result)
   ```
   Use `rec.valid_soil_types()` / `rec.valid_crop_types()` to see which
   categorical values the model was trained on.

## Irrigation Prediction module

Binary classification: predicts whether irrigation is needed given crop
type, days since planting, soil moisture/temperature, and weather
(temperature/humidity). See `data/irrigation/README.md` for dataset
size/imbalance caveats.

1. Dataset is already included at `data/irrigation/irrigation_prediction.csv`.
2. Train:
   ```bash
   python -m src.irrigation_prediction.train
   ```
3. Predict:
   ```bash
   python -m src.irrigation_prediction.predict
   ```
   Or import directly:
   ```python
   from src.irrigation_prediction.predict import IrrigationPredictor

   predictor = IrrigationPredictor()
   result = predictor.predict({
       "CropType": 1,  # 1=Paddy, 2=Ground Nuts
       "CropDays": 3, "Soil Moisture": 230, "Soil Temperature": 25,
       "Temperature": 30, "Humidity": 60,
   })
   print(result)  # {'irrigate': False, 'confidence': ..., 'probabilities': {...}}
   ```

## Disease Detection module

Multi-crop CNN: classifies a leaf photo into `<Crop>___<Disease-or-healthy>`.
**Important — this is a small 8-class subset of PlantVillage, not the
full 38-class dataset.** See `data/disease_images/README.md` for exactly
what's included and how to scale it up.

1. Images are already included at `data/disease_images/<ClassName>/*.jpg`.
2. Train:
   ```bash
   python -m src.disease_detection.train
   ```
3. Predict on a leaf image:
   ```bash
   python -m src.disease_detection.predict
   ```
   Or import directly:
   ```python
   from src.disease_detection.predict import DiseaseDetector

   detector = DiseaseDetector()
   result = detector.predict("path/to/leaf_photo.jpg")
   print(result)  # {'top_prediction': 'Tomato___Early_blight', 'crop': 'Tomato', ...}
   ```

## IoT Simulator module

**Synthetic by design** — the paper's IoT layer has no real deployment,
so this generates physically-plausible sensor readings (diurnal cycles,
gradual soil moisture depletion) rather than using real data. See
`src/iot/README.md` for details, including two real cross-dataset
inconsistencies (soil moisture units, crop naming) it surfaced.

```bash
python -m src.iot.simulator   # preview raw sensor readings
python -m src.iot.demo        # full pipeline: sensor -> 3 trained models -> report
```

## Frontend (Streamlit)

A simple browser UI to test all four models without curl/Swagger — upload
a leaf photo, or fill in number fields for the other three models, and
see results immediately. Includes an IoT Simulator tab that runs one
simulated sensor reading through crop, fertilizer, and irrigation at once.

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` in your browser automatically.

## API layer

FastAPI app tying all four trained models together, plus the IoT simulator.

```bash
uvicorn api.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` for interactive Swagger docs, or use:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Which models loaded successfully |
| `/predict/crop` | POST | Crop recommendation (JSON body) |
| `/predict/fertilizer` | POST | Fertilizer recommendation (JSON body) |
| `/predict/irrigation` | POST | Irrigation prediction (JSON body) |
| `/predict/disease` | POST | Disease detection (multipart image upload) |
| `/iot/simulate` | GET | One simulated sensor reading |
| `/iot/field-report` | POST | Simulated reading run through crop + fertilizer + irrigation models at once |

Example:
```bash
curl -X POST http://127.0.0.1:8000/predict/crop \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":20.8,"humidity":82.0,"ph":6.5,"rainfall":202.9}'
```

## Project structure

```
smart-agri-assistant/
├── config.py               # paths, hyperparameters
├── data/                    # datasets (not committed; see per-module READMEs)
├── models/                  # trained model artifacts
├── src/
│   ├── common/               # shared preprocessing + metrics utilities
│   ├── crop_recommendation/  # dataset.py, model.py, train.py, predict.py
│   ├── fertilizer_recommendation/  # Phase 2
│   ├── irrigation_prediction/      # Phase 3
│   ├── disease_detection/          # Phase 4
│   └── iot/                        # Phase 5 — sensor simulator
└── api/                      # Phase 6 — FastAPI serving layer
```

## Notes on scope

The source paper is a research-methodology document, not a detailed
software spec — it doesn't name specific datasets, a framework, or an
API surface. Assumptions made to make this buildable are documented in
the assistant's response where this project was scaffolded; the main
ones: PyTorch for all models, public Kaggle-schema datasets for the
tabular tasks, and a mocked/simulated IoT layer (since no real sensor
deployment exists yet per the paper itself).
