# Smart microscopy with a spiral survey

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prakhar-Dutta/ASAP/blob/main/tutorials/dna-imaging/smart-scanning/notebooks/DNA_simulation_and_spiral_scanning.ipynb)

The main notebook has two parts. **Part 1** prepares multi-molecule MD or
non-MD AFM frames, visualises initial and relaxed chains, compares with real
relaxed-DNA Scan 55, checks two generated frames, and exports datasets.
**Part 2** loads the manuscript's conditional diffusion or deterministic
U-Net checkpoint, or optionally trains a new model, then predicts and assesses
second-pass scan boxes on simulated fields and real or uploaded data.

## Folder layout

```text
smart-scanning/
├── notebooks/
│   └── DNA_simulation_and_spiral_scanning.ipynb
├── spiral_simulation_helpers.py
├── spiral_learning_helpers.py
├── setup_spiral.py
├── requirements.txt
├── checkpoints/
│   ├── diffusion/                 # original EMA, curves and provenance
│   ├── regression/                # original EMA, curves and provenance
│   └── user_runs/                 # optional new runs; ignored by Git
└── assets/
    ├── reference-fields/          # 40 original validation fields + one test field
    ├── scan55_predictions.npz     # saved original predictions for the shared scan
    └── manuscript_settings.json
```

All functions and classes are in the two Python helpers. Part 1 reuses the
chain generator, OpenMM force field and AFM renderer from
`../segmentation/dna_simulation_helpers.py`. Shared noise and the full native
filtered Scan 55 are in `../common-assets/`, with provenance JSON files.

## Setup and execution

On Colab, select a GPU for MD or live neural inference, then run **cell 0**.
It downloads individual files without cloning the repository: both original
EMA checkpoints (about 48 MB each), 17 MB of reference fields, training curves,
helpers and shared assets. Downloads are cached. The setup also installs
OpenMM and the Python 3.12 **SPM-only TopoStats profile** when selected.
The setup script is refreshed each run so an older cached installer is replaced.

Local setup:

```bash
pip install -r requirements.txt
pip install openmm                              # MD only
python ../segmentation/setup_topostats.py --spm-only  # your own SPM files
```

Then open the notebook from `notebooks/`. Use your existing Jupyter environment
or install JupyterLab separately. Dependencies and execution are checked in
Python 3.12.13 locally; the hosted Colab runtime is not used for validation.

The spiral notebook only calls `topostats.io.LoadScans` and
`topostats.filters.Filters`. The SPM-only profile reuses the segmentation
tutorial's verified upstream wheel and Python compatibility changes, but
installs only dependencies reached by those APIs. It omits TensorFlow/Keras,
skan and topoly, and does not install the full TopoStats command-line tool.
It leaves scan reading, calibration and filtering code unchanged. The regular
segmentation installer without `--spm-only` still builds its full contour
profile. Bundled Scan 55 and NPY inputs need no TopoStats installation.

| Goal | Sections |
|---|---|
| Colab setup | 0 first |
| Part 1 previews | 1.1 → 1.2 → 1.3 → 2 → 3 → 4 → 5 → 6 |
| Dataset export | 1.1 → 1.2 → 1.3 → 7; enable `GENERATE_DATASET` |
| Use manuscript checkpoints | Part 2, 8 → 15 |
| Train on your own generated data | Export in 7, then 8 → 15 with `TRAIN_NEW_MODEL = True` |

Dataset export is disabled by default so running all previews does not start
a 1000-frame MD job. Checkpoints are loaded by default; training is optional.

## Part 1 controls

* `(1, 3)` samples 1, 2 or 3 molecules uniformly. `(2, 4)` samples 2, 3 or 4;
  `[2, 4]` samples only 2 or 4, and duplicate entries act as weights.
* The chain schematic displays the rounded mean count in different colours;
  `(2, 4)` shows three chains. The MD animation uses one chain.
* Default frame: square, 640 × 640 nm. Choose `FRAME_SHAPE = "rectangle"`
  to use both width and height. Dimensions must fit an integer model grid.
* Fine composition pixels are 1.25 nm; model pixels are 2.5 nm. The default
  images are 512 × 512 and 256 × 256 pixels over the same physical field.
* Long-DNA contour: 768.4 nm. Defaults use the optimised MD force field,
  350 recorded frames × 300 steps, 1.8–2.8 nm strand heights and 6–8 nm tips.
  The non-MD option uses the calibrated geometric generator and renderer.
* The noise cell has the segmentation notebook's blank-SPM and PSD controls.
  Its default selects the original MD single-blank spectrum; measured blank
  patches and the shared ensemble PSD are also selectable. The spectrum is
  sampled at physical frequencies; one background is added to each frame.
* The first molecule remains complete. Extra molecules may cross the frame
  edge when at least 30% of their mask remains visible. Clearance is 10 nm.
  Set `ALLOW_PARTIAL_MOLECULES = False` to keep every molecule complete.
  Impossible packing raises rather than silently dropping molecules.

The historical experiment composited cached MD source scenes with variable
220–609 nm fields, centred in a 640-nm canvas. Part 1 generates new independent
chains on a configurable frame using the shared physical recipes. Its new
exports are distinct from the supplied original reference fields. The nominal
DNA target width is 6.87 nm (4.47 nm plus a 1.2-nm extension on each side).

Exported datasets contain `manifest.csv`, `dataset_config.json`, and
`frames/*.npz`/`*.json`. Arrays include clean/noisy heights, noise, instance
labels, DNA/crossing masks, model-grid arrays and full chain coordinates.
They require no pickle. JSON records contours, seeds and visible fractions.
Splits default to 80/10/10; molecule seeds remain distinct across splits.
Completed records are flushed during export, and existing data are protected.

## Part 2 defaults and original weights

`MODEL_KIND` selects `diffusion` or `regression`. Both use the same 11,965,058
parameter conditional residual U-Net: base width 64, widths 64/128/192/256,
GroupNorm, noise embedding and bottleneck attention. Supplied files contain
the **unchanged original EMA tensors**, with optimizer/training copies removed.
The selected states are diffusion step 23,000 and regression step 10,000.
Their JSON files record source checksums and training history.

DeepTrack Sources hold manifest rows, and lazy pipelines read arrays and
construct observations only when requested. New training uses the Part 1
manifest. Regression minimises hidden-height MSE plus in-frame DNA BCE.
Diffusion uses weighted EDM reconstruction plus low-noise soft-clDice (0.05)
and relative skeleton-length (0.02) terms. The notebook explains the equations,
masking, thresholds and validation metrics.

Optional training defaults match the manuscript recipe: 30,000 diffusion or
10,000 regression updates, batch 12, AdamW at 2e-4, weight decay 1e-4, 1000-step
warm-up, cosine decay and EMA approaching 0.999. The original loss-validation
set had 64 fixed examples. The separate 40 ROI-validation fields selected the
scan-box thresholds; the supplied checkpoint and field caches reproduce those
predictions without retuning on real data.

## Scan-box strategy

The outward spiral has 12 turns and records heights every 5 nm. Diffusion uses
24 denoising steps and **all 16 unfiltered configurations**. Pixels with
occupancy **≥2/16** are joined to observed DNA anchors, and each eight-connected
region containing an anchor gets a rectangle. The deterministic cutoff is
**0.05**. Both use zero additional margin. Nested boxes are removed.

A molecule is covered when the union of boxes contains ≥95% of its reference
mask. Charged saving subtracts observed survey pixels and the sum of all box
areas, so repeated visits count. Oversized plans use a full-frame second pass
and still charge the survey. These are area estimates, not hardware timings.

Matching reference caches are enabled by default for a fast CPU run and exact
saved experiment predictions. They are accepted only for the original weight
checksum and matching observations, frame and sampler settings. Disable
`USE_CACHED_REFERENCE_PREDICTIONS` for live inference. New models, altered
observations and uploaded files run the network directly; live diffusion is
batched to limit memory.

The simulated prediction preview shows three panels: acquired DNA anchors,
DNA occupancy likelihood (U-Net probability in regression mode), and the
second-pass boxes. Individual generated configurations are not displayed.

## Real Scan 55 and user data

The canonical shared Scan 55 is in
`../common-assets/example-frames/relaxed_scan55.npz`: 512 × 512 native pixels
at 1.190185546875 nm/pixel, spanning 609.375 × 609.375 nm. Part 1 displays it
at that size. Part 2 uses the same physical resampling and centre padding as
the original experiment. Its three automatic TopoStats grain proposals are
assessment references, not manually annotated truth or model input.

For your own data, choose a local path or Colab upload and select SPM or NPY.
NPY heights must be in nm; expose their pixel calibration when known. SPM files
are loaded and filtered with TopoStats; channel, YAML and Gaussian size are
configurable. A frame-size mismatch triggers the requested resize and warning
that performance may be sub-optimal. Original extents are retained for mapping
boxes back to source coordinates. Unknown NPY calibration is reported.
Without reference masks, Dice/IoU and molecular coverage are not scored.

Real-data analysis compares the learned plan with DBSCAN and true 2-D linear
interpolation on the same survey. Defaults are frozen from the manuscript's
`A_sim_val`, faithful `outward_12`, capped selection. DBSCAN uses a 90-nm
neighbourhood, one minimum sample and a 40-nm box margin. Interpolation uses
all measured heights, fills outside their convex hull with the measured
median, thresholds at 1.25 times the anchor-height threshold and adds a 40-nm
margin to each connected region. Controls are exposed in cell 8. Neither
baseline uses hidden heights or reference labels to choose boxes.

The figure shows anchors, likelihood, learned/DBSCAN/interpolation boxes and
a three-method savings chart. All plans share survey cost, overlap charging,
nested-box removal and full-frame fallback. Coverage counts are displayed
when references exist; smaller boxes may miss molecules. Negative savings
are retained and mean the survey plus second pass costs more area than one
full-frame scan. This comparison estimates acquired area, not hardware time.

## Validation

```bash
python -m unittest discover -s tests -p "test_spiral_*.py" -v
```

Checks cover frame calibration/counts, placement, noise, export, lazy loading,
both objectives and short training runs, live sampling, hidden-data isolation,
the original 40-field scan-box result, Scan 55, and NPY resize/coordinate mapping.
Both helper files are formatted and checked for PEP8 with 79-character lines.
The complete notebook is executed with its defaults and with the alternative
non-MD, rectangular, 2–4-molecule setting and regression model. Additional
checks cover SPM loading and live original-checkpoint inference.
Exact execution configurations and results are recorded in
[`validation.json`](validation.json).

## Further resources and citation

[TopoStats](https://github.com/AFM-SPM/TopoStats) provides advanced biomolecular
tracing and topology analysis. [DeepTrackAI](https://github.com/DeepTrackAI)
provides simulation and deep-learning workflows for microscopy.

Prakhar Dutta, Jiacheng Huang, Nazli Demirpehlivan, et al. "ASAP (AFM simulation
and analysis pipeline): a unified deep learning framework for simulation,
quality control, and segmentation in atomic force microscopy", Proc. SPIE
14198, Emerging Topics in Artificial Intelligence (ETAI) 2026, 1419804
(9 Oct 2026). [https://doi.org/10.1117/12.3113626](https://doi.org/10.1117/12.3113626)
