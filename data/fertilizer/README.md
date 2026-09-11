# Fertilizer Recommendation dataset

`fertilizer_prediction.csv` in this folder is the real public "Fertilizer
Prediction" dataset (99 rows, 7 fertilizer classes, 5 soil types, 11 crop
types, no missing values). Columns:

```
Temparature, Humidity, Moisture, Soil Type, Crop Type,
Nitrogen, Potassium, Phosphorous, Fertilizer Name
```

(Note the dataset's own spelling: "Temparature", not "Temperature" — kept
as-is to match the source data; `dataset.py` handles it consistently.)

## Known limitation

This is a small dataset — 99 rows total, and some fertilizer classes have
very few examples (as few as 7). Validated model performance: **95% test
accuracy**, but macro precision/recall are lower (~0.79/0.86) because a
single misclassified sample in a rare class swings those metrics hard on
a 20-sample test set. Treat this model as a reasonable baseline
demonstration of the pipeline, not a production-grade fertilizer
recommender — a real deployment would need a substantially larger,
more balanced dataset.
