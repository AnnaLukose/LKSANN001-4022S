# SnoCam+ image-processing pipeline for sinking marine-snow particles

An end-to-end pipeline that takes time-lapse images of particles settling onto a trap plate and turns them into a
particulate organic carbon (POC) flux: it removes the background, finds every particle, counts each new arrival
once, classifies it, and converts its size and type into carbon. Because real imagery with known particle counts is
not available, the pipeline is tested on **synthetic image sequences with a known ground truth** (also generated
here), and every stage is scored against that truth.

## Background

Sinking particles (aggregates, faecal pellets and detritus, collectively "marine snow") carry carbon from the
surface ocean to depth and drive the biological carbon pump. Sediment traps and Marine Snow Catchers capture these
particles, but they average over the whole deployment and need slow manual microscopy. Trap-mounted cameras such as
GelCam (Song et al., 2025) and SnoCam+ (Edirisinghe et al., 2024) photograph particles as they land, which gives
flux and composition at much finer time resolution.

---

## How the pipeline works

```
Gel-trap archive (real particle crops, classified by an expert)
        │
        ▼
 1 config ─► 2 library ─► 3 patterns ─► 4 render ──► raw/   (synthetic frames)
                                              └────► truth/ (outline, class, size and carbon of every particle)
                                                         │
 raw/ ─► 5 preprocess ─► 6 segment ─► 7 track ─► 8 tables ─► 9 classify ─► 10 carbon + flux
                                                                                  │
                                         truth/ ───────────────────────────► scored at every stage
```

Steps 1-4 build the synthetic data once. Steps 5-10 run on every sequence. The comparison notebooks
(`piv_compare`, `cls_compare`, `cls_compare2`) are the evaluation methods that decided what Steps 6, 7 and 9 use.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── pipeline.ipynb
│   ├── piv_compare.ipynb
│   ├── cls_compare.ipynb
│   └── cls_compare2.ipynb
└── src/
    └── (the Python files listed below)
```

---

## Getting started

The notebooks are written for **Google Colab** with the code and data kept in **Google Drive**. The classification
steps need a GPU runtime.

### 1. Download the repository

```bash
git clone https://github.com/AnnaLukose/LKSANN001-4022S.git
```

or use **Code → Download ZIP** on GitHub and unzip it.

### 2. Put it in Google Drive

By default all the notebooks look for everything in one Drive folder, `MyDrive/snocam_pipeline/`. Upload the
repository there so that the layout is:

```
MyDrive/snocam_pipeline/
    src/            (all the .py files, including __init__.py)
    notebooks/
```

If you want a different location, change the `DRIVE` path in the **Setup** cell of each notebook.

### 3. Open a notebook in Colab

Open the notebook, then:

- *Runtime → Change runtime type → T4/L4 GPU
- Run the cells in order: **Settings → Setup → Status**, then the Parts you need

Every notebook is resumable. Each finished sequence, model and result is saved to Drive straight away and skipped
the next time, so a Colab disconnect only loses the step that was running.

### 4. Run order

| Order | Notebook | What it does |
|---|---|---|
| 1 | `pipeline.ipynb` | Builds the sprite library and the synthetic data, then runs Steps 5–10 and scores every stage |
| 2 | `piv_compare.ipynb` | **Part 0** trains the hybrid segmenter's quality filter (shared by notebooks 2–4); Parts A–E compare bulk-motion methods and trackers, including a large-motion stress test |
| 3 | `cls_compare.ipynb` | **Part A** builds the particle tables reused by notebook 4; then Experiment 1 (feature models and CNNs) and the resolution test |
| 4 | `cls_compare2.ipynb` | Experiment 2: prior-shift correction, description features and combined models |

Notebook 1 must run first (at least Part A), because the other three read its `run_13um` data.

### 5. Input data

The only external data are the expert-labelled oblique gel-trap particle crops of Durkin et al. (2021), published on
BCO-DMO (dataset 860725). `pipeline.ipynb` **Part A** downloads them and builds the sprite library automatically.
Everything else is generated.

### 6. Requirements

Colab already has most of these; the notebooks install the rest (`timm`, `openpiv`, `lightgbm`, `xgboost`,
`openpyxl`) in their Setup cell.

```
numpy
pandas
scipy
scikit-image
scikit-learn
opencv-python
matplotlib
joblib
openpyxl
torch
timm
openpiv
lightgbm
xgboost
```

---

## What each Python file does

### Generator (Steps 1-4, done once)

| File | Description | Used in (.ipynb) |
|---|---|---|
| `config.py` | **Step 1.** The single place the pixel scale (13 µm/px) is set. Sets the frame size, the smallest counted particle (50 px) and the classifier crop size, and converts every pixel setting so it keeps the same physical size. Also holds the particle classes (the eight classifier classes and the merged five-class view). | `pipeline` |
| `library.py` | **Step 2.** Turns the expert-labelled microscope crops into sprites: cuts each particle out with an opacity mask, records its true size in µm, and splits the library into train / validation / test by micrograph so a field of view never appears in two splits. | `pipeline` |
| `patterns.py` | **Step 3.** The scene patterns: the class mix of each real trap sample, so different sequences see different mixes of particle types. | `pipeline` |
| `render.py` | **Step 4.** The synthetic camera. Builds the empty instrument (optical window, LED ring, vignetting), then for each of the 30 frames moves the trap, adds new arrivals, draws each particle at its true size, adds daylight, haze and dust, and applies blur, noise, the Bayer mosaic and JPEG. Writes the raw frames and the full ground truth. | `pipeline` |
| `excel.py` | Helper that writes the multi-sheet `truth.xlsx` workbooks. | `pipeline` (via `render.py`) |

### Pipeline (Steps 5-10, run on every sequence)

| File | Description | Used in (.ipynb) |
|---|---|---|
| `preprocess.py` | **Step 5.** Background removal following Song et al. (2025): keeps the red channel, applies the static mask and removes the static background with proper orthogonal decomposition (POD). | `pipeline` |
| `segment.py` | **Step 6.** The hybrid segmenter (`hybrid_segment`): several candidate outlines per blob, a gradient-boosted quality filter that scores each one, and the best non-overlapping candidates above the cut-off τ = 0.30, with a size correction learned on the training sequences. Scores recall, precision, carbon found and size against the truth (IoU ≥ 0.5). | `pipeline` |
| `track.py` | **Step 7.** Fits the gel's bulk motion between frames with RANSAC, then links particles with Song et al.'s cross-correlation tracker, with a search window that widens when a particle is not found (`tracking`). A track that starts after the first frame is a new arrival. Scores links, arrivals and re-counts against the truth. | `pipeline` |
| `tables.py` | **Step 8.** For every detection: size, shape and brightness measurements, its track, an 80 px RGB crop and its truth (the true particle under it, or "false detection"). | `pipeline` |
| `classify.py` | **Step 9.** The ResNet-18 classifier and everything around it: prior-free scores, averaging over each track's first detections, the decision rule, the size gate and the scoring (accuracy, balanced accuracy, precision / recall, confusion, carbon budget). | `pipeline` |
| `hitl.py` | **Step 9.** The human-in-the-loop procedure of Amaral and Durkin (2025): trains the out-of-domain ensemble, then for each budget shows the expert (simulated by the ground truth) the most confident particles per class (+verify) and extra rare-class particles (+rareboost), fine-tuning after each. | `pipeline` |
| `carbon.py` | **Step 10.** Carbon per particle, C = A·V^B, with the shape and constants of each class from Durkin et al. (2021). Gives both the true and the predicted carbon. | `pipeline` |
| `flux.py` | **Step 10.** Cumulative carbon through each deployment and flux per sequence and per class, from the pipeline and from the truth. Also Song's white-pixel area method for comparison. | `pipeline` |
| `evaluate.py` | Checks the generator and background removal against the truth, and builds one summary table of every stage. | `pipeline` |
| `figures.py` | All figures: the Song et al. and Amaral and Durkin style figures, plus the stage checks (detection boxes, τ curve, effort curve, confusion matrix, cumulative carbon). | `pipeline` |
| `__init__.py` | Makes `src/` a package so the notebook can import it. | `pipeline` |

### Evaluation methods (done once)

| File | Description | Used in (.ipynb) |
|---|---|---|
| `scale.py` | Reads the pixel scale from the data (`truth/config.json`) and converts every pixel setting of the comparison scripts so it keeps its physical size. | `piv_compare`, `cls_compare`, `cls_compare2` |
| `hybrid_segment.py` | The hybrid segmenter used to produce detections for the tracking and classification comparisons. Trains the quality filter the first time and saves it with cached detections, so the other notebooks reuse the same detections. | `piv_compare` (and `cls_compare`, `cls_compare2` through `piv_compare.py`) |
| `piv_compare.py` | Compares bulk-motion methods (Song's PIV, OpenPIV, optical flow and RANSAC) and trackers (Song's tracker, `tracking`, nearest neighbour, Hungarian assignment and others): motion error, link precision and recall, arrivals counted correctly and re-counts, plus a large-motion stress test. This is where RANSAC + `tracking` was chosen. | `piv_compare`, `cls_compare`, `cls_compare2` |
| `cls_compare.py` | Experiment 1: compares feature-based models and CNNs on the particles the pipeline finds. Includes the resolution test, which splits the results by particle size in µm and in pixels. | `cls_compare` |
| `cls_compare2.py` | Experiment 2: the most promising classifiers, corrected and combined, with the prior-shift correction, description features based on Durkin's and Song's class definitions, and the λ trade-off between accuracy and balanced accuracy. | `cls_compare2` |

---

## Acknowledgements

Supervisor: Robyn A Verrinder (UCT). Co-supervisor: Dr S Thomalla (CSIR SOCCO).
Particle crops from Durkin et al. (2021): [https://www.bco-dmo.org/dataset/860725](https://www.bco-dmo.org/dataset/860725);
microscope calibration from Durkin's processing code: [https://github.com/cadurkin/Sediment_Trap_Gel_Image_Processing_2020](https://github.com/cadurkin/Sediment_Trap_Gel_Image_Processing_2020).
Background removal and tracking adapted from Song et al.'s GelCam code: [https://github.com/EE-Shuan-Song/GelCam_particle_tracking](https://github.com/EE-Shuan-Song/GelCam_particle_tracking).
