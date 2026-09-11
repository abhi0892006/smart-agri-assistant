# Disease Detection dataset

Images here are a real, curated subset of the public **PlantVillage**
dataset (https://github.com/spMohanty/PlantVillage-Dataset — 54,306
images, 14 crop species, 26 diseases; introduced in Mohanty et al. 2016,
"Using Deep Learning for Image-Based Plant Disease Detection").

## What's actually included here

**8 classes across 4 crops, 150 images each (1200 total):**

- `Apple___Apple_scab`, `Apple___healthy`
- `Corn_(maize)___Common_rust_`, `Corn_(maize)___healthy`
- `Potato___Early_blight`, `Potato___healthy`
- `Tomato___Early_blight`, `Tomato___healthy`

This is a **deliberate, disclosed subset** — not the full dataset. The
full PlantVillage dataset has 38 classes across 14 crops and would take
far longer to train (and much more disk/compute) than is practical for
this pipeline demonstration. Images were taken from the front of each
class's folder (not randomly shuffled), so this is a consistent,
reproducible subset, not a random sample.

## Validated performance

93.9% test accuracy, 0.939 macro F1, on a held-out 15% test split (180
images) — genuinely good, though earned on an easier problem than full
multi-crop disease detection: only 4 crops, 1 disease type per crop (plus
healthy), and no cross-crop visual confusion between diseases that look
similar across species.

## Scaling this up

To move toward what the paper actually describes:
1. Add more classes from the full PlantVillage class list (see
   `config.DISEASE_CLASSES` — the training pipeline already generalizes
   to any subset of those 38 folder names).
2. Increase images per class (the current cap of 150/class is a speed
   choice for CPU training, not a data limitation).
3. Swap the from-scratch CNN in `model.py` for a pretrained backbone
   (ResNet18/VGG16, as the paper's cited literature does) via transfer
   learning — the input normalization here already uses ImageNet
   mean/std specifically to make that swap easy later.
4. Use a GPU — this dataset size was chosen specifically to make CPU
   training feasible in a few minutes; the current small custom CNN
   won't scale well to hundreds of classes on CPU.
