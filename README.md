# C.-elegans-Connectomics
*C. elegans* connectomics is the study of the worm’s nervous system by mapping every neuron and the synapses connecting them, turning microscope images into a complete wiring diagram, then using that map to explain how specific circuits generate behaviour, pipeline for volume EM cutouts, from image stacks to a draft connectivity graph.

Reconstructing neural circuits from volume electron microscopy (vEM) is a central route to mechanistic neuroscience, but the full workflow, data ingestion, alignment, segmentation, synapse detection, graph construction, and iterative proofreading, is often presented as a collection of large-scale tools that are difficult to run end-to-end on modest hardware. Here, a lightweight proof-of-concept (PoC) connectomics pipeline is presented, designed to run on a CPU-only personal computer while preserving the conceptual stages used in production connectomics. Using a small public vEM cutout from a *Caenorhabditis elegans* larval dataset hosted in BossDB, I implement, (i) data retrieval, (ii) denoising and local contrast enhancement, (iii) per-slice translational alignment, (iv) neurite instance segmentation via a watershed-based baseline with 3D label stitching by overlap, (v) heuristic synapse candidate detection, (vi) conversion into a directed, weighted connectivity graph, and (vii) a human-in-the-loop proofreading interface concept in which corrections are logged for model improvement. On a 256 × 256 × 64 voxel cutout, the pipeline generates neurite labels (maximum instance ID 6779 across the volume), 12,297 synapse candidates (155 on the mid-volume slice), and a draft connectivity graph containing 5,244 nodes and 7,577 directed edges at a conservative candidate threshold. While the PoC does not aim to match deep learning performance in modern connectomics systems that requires large data, it provides a transparent, reproducible scaffold that mirrors the real connectomics lifecycle, enabling rapid experimentation, teaching, and iterative engineering prior to scaling.

A lightweight, end-to-end **connectomics pipeline** that runs on a **CPU-only Mac laptops**, starting from a small public ***C. elegans* EM cutout**, then **preprocesses and aligns** slices, performs **baseline neurite segmentation**, detects **synapse candidates** (heuristic), builds a **wiring graph**, and outputs **figures and tables**.

Dataset source: BossDB, Mulcahy 2022 *C. elegans* EM.  
BossDB project page: https://bossdb.org/project/mulcahy2022

> This repository is a proof-of-concept, it demonstrates the workflow structure and reproducible outputs on small cutouts, it is not a production-grade connectome reconstruction system.

## What this repo does

Pipeline stages implemented:

1. **Collect data**, download a small 3D EM cutout from BossDB (no huge downloads).
2. **Clean and align**, denoise + contrast normalize, then rigid translation alignment across slices.
3. **Run AI segmentation (baseline)**, slice-wise watershed + simple 3D stitching to create neurite instance IDs.
4. **Detect synapses (heuristic)**, blob-like dark features near boundaries between segments.
5. **Build wiring map**, convert candidate synapses into a directed weighted graph.
6. **Proofread and improve (optional)**, Napari proofreading, then refinement via RandomForest or CPU U-Net.
7. **Keep it running**, minimal run logging to SQLite, consistent outputs, reproducible settings.
8. **Publish**, auto-generated figures (`.pdf`) and tables (`.csv`) for reports/manuscripts.

## Repository structure

```text
├── run_pipeline.py
├── napari_proofread.py                  # optional, human-in-the-loop proofreading
├── train_refine_rf.py                   # optional, RF refinement from corrected slices
├── train_unet2d_selftrain.py            # optional, small CPU 2D U-Net self-training
├── rebuild_graph_and_figures.py         # recompute synapses + graph from refined labels
└── outputs/
    ├── data/
    ├── figures/
    ├── tables/
    └── runs/
```
# Model Card, *C. elegans* Connectomics Proof-of-Concept Models

## Model overview
This repository contains **proof-of-concept models** used in a CPU-only pipeline to convert a small public ***C. elegans* electron microscopy (EM) cutout** into (i) **neurite/instance segmentation labels**, (ii) **synapse candidate detections** (heuristic), and (iii) a **connectivity graph**. The models are designed to be lightweight, reproducible, and runnable on a consumer CPU machine.

The repository supports three modelling components:

1. **Baseline neurite instance segmentation (classical CV)**  
   A slice-wise **foreground detection + watershed instance segmentation** followed by simple **3D stitching by overlap**.

2. **Optional refinement, RandomForest (supervised from human corrections)**  
   A **RandomForestClassifier** learns to classify pixels as **neurite vs background** using a small set of proofreading-corrected slices. The refined neurite probability maps are converted back into instances via watershed, then stitched in 3D.

3. **Optional refinement, small 2D U-Net (self-training, CPU)**  
   A compact **2D U-Net** is trained on patches using pseudo-labels from the baseline segmentation plus higher-confidence corrected slices (if available). Slice-wise probabilities are converted to instances, then stitched in 3D.

> Important, this is a proof-of-concept implementation intended to demonstrate end-to-end workflow mechanics on small data cutouts, not a validated production connectomics system.

## Intended use
**Primary intended use**
- Demonstrate a reproducible connectomics workflow, from EM cutout to segmentation to a graph representation.
- Provide a CPU-only baseline that can be extended to stronger ML approaches.
- Provide a human-in-the-loop pathway (Napari proofreading → model refinement).

**Users**
- Researchers learning connectomics pipelines.
- Developers prototyping image-to-graph workflows.
- Students exploring segmentation, proofreading, and graph construction.

**Out-of-scope use**
- Producing publication-quality connectomes without curated labels, extensive validation, and expert proofreading.
- Inferring true synaptic directionality or neuron identities.
- Clinical, diagnostic, or safety-critical applications.

## Model details

### A. Baseline segmentation (classical CV)
**Type**
- Watershed instance segmentation per slice.

**Inputs**
- Aligned 2D EM slices (float32 in [0,1]).

**Outputs**
- Per-slice instance labels, then a 3D label volume after overlap stitching.

**Key assumptions**
- Foreground can be approximated by intensity quantiles after preprocessing.
- Instances can be split using distance-based seeds and watershed.
- Instances align across adjacent slices via pixel overlap.

### B. RandomForest refinement (optional)
**Type**
- RandomForest pixel classifier, binary neurite vs background.

**Training data**
- A small set of human-corrected slices created by Napari proofreading.

**Features**
- Raw intensity, Gaussian-blurred versions, edge magnitude (Sobel), local variance proxy.

**Output**
- Neurite probability maps, converted to instances via watershed.

**Strengths**
- Very CPU-friendly, fast iteration.
- Works with limited corrections.

**Limitations**
- Handcrafted features may not generalize across imaging conditions.
- Sensitive to segmentation and labeling style.

### C. Small 2D U-Net self-training (optional)
**Type**
- Lightweight 2D U-Net trained on patches.

**Training data**
- Pseudo-labels from baseline segmentation + (optional) corrected slices.

**Loss**
- Weighted combination of BCEWithLogitsLoss and Dice loss.

**Output**
- Slice-wise neurite probability maps, converted to instances via watershed.

**Strengths**
- Can learn richer features than RandomForest.
- Still runnable on CPU if kept small.

**Limitations**
- Self-training can reinforce baseline errors.
- Requires careful patch sizing and training settings on low-memory machines.

## Training data and provenance
**Dataset source**
- BossDB, Mulcahy 2022 C. elegans EM dataset (downloaded as small cutouts at run time).

**Data access**
- The pipeline uses BossDB APIs to download only a small volume cutout (e.g., 192×192×48 or 256×256×64 voxels).

**Human annotations**
- Optional proofreading annotations are created by the user in Napari and stored locally as corrected label volumes.

**No dataset redistribution**
- This repository does not ship EM images, it downloads cutouts directly from the public source.

## Evaluation

### What is evaluated in this proof-of-concept
- Internal diagnostic metrics for refinement models, for example validation AUC for the RandomForest on held-out corrected pixels.
- Sanity-check figures (overlays, alignment shifts, graph statistics).

### What is not evaluated here (and would be required for scientific claims)
- Standard connectomics segmentation metrics (e.g., variation of information, adapted Rand error) on curated ground truth.
- Synapse detection precision/recall against annotated synapses.
- Neuron identity correctness or connectivity accuracy vs a reference connectome.

## Ethical considerations
- The pipeline uses public neuroscience imaging data and does not involve human subject data.
- The pipeline can produce misleading graphs if interpreted as true synaptic wiring without validation.
- Users should treat outputs as exploratory, requiring expert proofreading and ground-truth benchmarking.

## Biases and failure modes
Common failure modes include:

- **Over-segmentation** (one neurite split into many fragments).
- **Under-segmentation** (multiple neurites merged into a single instance).
- **Slice alignment drift** (small errors accumulate across Z).
- **Heuristic synapse artifacts** (dark blobs are not always synapses).
- **Pseudo-label reinforcement** (U-Net may learn baseline mistakes in self-training).

Mitigations in this repo:
- Keep volumes small and iterate.
- Add proofreading on representative slices.
- Prefer RandomForest refinement for fast, controlled improvements.
- Use figures to inspect results before trusting downstream graph outputs.

## Compute and hardware requirements
Designed for:
- CPU-only macOS machines, including Mac mini M2 with 8 GB RAM (using safe presets).

Recommended runtime settings:
- `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`
- Use `--preset safe8gb`, `--save_minimal`, and `--aligned_dtype float16` to reduce resource use.

## How to use (high-level)
Baseline run:
```bash
python run_pipeline.py --preset safe8gb --save_minimal --aligned_dtype float16
```
# Dataset Datasheet, *C. elegans* Connectomics Proof-of-Concept (BossDB Mulcahy 2022 Cutouts)

## 1. Dataset summary
This repository uses **small cutouts** from a publicly hosted ***C. elegans* electron microscopy (EM)** dataset served via **BossDB**. The pipeline downloads only a small 3D subvolume at runtime (for example 192×192×48 or 256×256×64 voxels) to support **CPU-only** proof-of-concept connectomics workflows on consumer hardware.

This datasheet describes:
- the upstream dataset source,
- how cutouts are selected and stored locally,
- derived artefacts produced by the pipeline (aligned volumes, segmentations, synapse candidate tables, graphs),
- limitations and appropriate use.

> This repository does not redistribute EM images, it downloads cutouts directly from the public source.

## 2. Motivation
**Why this dataset is used**
- Provide a real-world EM volume suitable for demonstrating an end-to-end pipeline from images → segmentation → synapse candidates → connectivity graph.
- Avoid very large downloads and storage burdens by using small cutouts.
- Enable reproducible experiments where the same dataset URI and cutout size can be logged and rerun.

**Intended users**
- Researchers and students learning connectomics pipeline mechanics.
- Developers prototyping image-to-graph workflows on limited hardware.

## 3. Dataset composition

### 3.1 Upstream dataset (public source)
**Source**
- BossDB project: Mulcahy 2022 C. elegans EM  
  Project page: https://bossdb.org/project/mulcahy2022

**Data type**
- Volume electron microscopy (grayscale intensity image stacks)

**Unit of data**
- Voxel intensity values (stored as 8-bit integers or rescaled to 8-bit for this PoC)

**Dimensionality**
- 3D volumes with axes (Z, Y, X), where Z is the slice index

### 3.2 Local cutouts downloaded by this repo
**Selection**
- A small rectangular subvolume (“cutout”) is downloaded from the dataset URI at runtime.
- A random jitter (controlled by `--seed`) is applied around a central location to avoid repeatedly sampling exactly the same region.

**Typical cutout sizes**
- `--preset safe8gb`: (X,Y,Z) = (192, 192, 48)
- `--preset medium`: (X,Y,Z) = (256, 256, 48)
- user-configurable: `--cutout X Y Z` (use caution on low-RAM machines)

**Local storage format**
- Numpy arrays saved as `.npy` files

**Local files created**
- `outputs/data/em_aligned_float16_zyx.npy` or `em_aligned_float32_zyx.npy`
- optional intermediates (if not using `--save_minimal`):
  - `outputs/data/em_cutout_u8_zyx.npy`
  - `outputs/data/em_preprocessed_f32_zyx.npy`

## 4. Data collection process

### 4.1 How data is obtained
- Data is downloaded on demand from BossDB using the `intern` library.
- Only the requested cutout range is fetched, avoiding large downloads.

### 4.2 Licensing and access
- The dataset is accessed from a public repository (BossDB).
- Users should consult the BossDB project page for licensing and usage terms.
- This repo does not bundle any dataset content.

## 5. Data preprocessing

### 5.1 Preprocessing steps applied in `run_pipeline.py`
For each 2D slice:
1. **Denoising** via a light Gaussian filter.
2. **Contrast normalization** using CLAHE (adaptive histogram equalization).

Purpose:
- Improve boundary visibility and reduce noise for downstream segmentation.

### 5.2 Alignment
- A rigid translation alignment is applied across slices using phase cross-correlation.
- Optional simulated misalignment can be applied for demonstration, then corrected.

Outputs:
- `outputs/tables/alignment_shifts.json`, per-slice estimated translations
- aligned volume stored as float16 or float32

## 6. Derived labels and annotations

### 6.1 Baseline neurite instance labels (derived)
- The pipeline produces neurite “instance” labels as a 3D integer array:
  - `outputs/data/neurite_labels_i32_zyx.npy`

Method:
- Slice-wise watershed segmentation, then 3D stitching by overlap.

Interpretation:
- Each integer ID represents a neurite fragment or neurite-like region, not a biologically validated neuron identity.

### 6.2 Optional human proofreading (user-generated)
- Users can correct segmentation errors on selected slices using Napari.
- These corrections are stored locally and used for refinement model training.

Typical files (implementation-dependent):
- `outputs/data/neurite_labels_corrected_i32_zyx.npy`
- `outputs/tables/proofread_slices.json`

### 6.3 Synapse candidate detections (derived, heuristic)
- The pipeline detects synapse-like candidates near segment boundaries:
  - `outputs/tables/synapse_candidates.csv`

Each candidate includes:
- `(z, y, x)` coordinates
- `pre_id`, `post_id` (heuristic direction)
- `score` (contrast-based proxy confidence)

Important:
- These are not validated synapses, they are heuristic candidates for proof-of-concept graph building.

## 7. Connectivity graph artefacts

### 7.1 Graph construction
- Candidates above a score threshold are converted into a directed weighted graph:
  - nodes: neurite instance IDs
  - edges: candidate synapses between instances
  - weights: number of candidate events for each ordered pair

### 7.2 Graph outputs
- `outputs/tables/graph_node_stats.csv`, per-node summary statistics
- `outputs/figures/Figure3_graph.pdf`, visual network layout

Interpretation:
- The graph is a workflow artefact representing candidate connectivity between segmentation instances, not a validated connectome.

## 8. Recommended uses

Appropriate uses:
- Pipeline prototyping and reproducibility exercises
- Demonstrations of image stack alignment, segmentation, and image-to-graph workflows
- Educational purposes and method development on small volumes

Not appropriate uses:
- Claims about true neuron identity, wiring, or synaptic directionality
- Biological conclusions without ground-truth annotations and extensive validation
- Production-scale connectome reconstruction without proper infrastructure

## 9. Known limitations

### 9.1 Data limitations
- Only a small cutout is used, it may not contain representative tissue features.
- Sampling location can affect segmentation and synapse heuristics substantially.

### 9.2 Label limitations
- Baseline segmentation is a classical CV approach, it can under- or over-segment neurites.
- Stitching across Z uses overlap heuristics, which can fragment or incorrectly merge instances.

### 9.3 Synapse candidate limitations
- Heuristic detection can produce false positives (dark blobs that are not synapses).
- The directionality assignment is heuristic and should not be interpreted biologically.

## 10. Potential sources of bias
- Cutout selection around the dataset center plus random jitter may bias toward certain anatomical regions.
- Intensity-based segmentation can bias toward structures with particular contrast profiles.
- Self-training refinement can reinforce baseline errors if proofreading coverage is small.

Mitigations:
- Log run parameters and seeds for reproducibility.
- Use multiple cutouts from different regions (different seeds and offsets).
- Add proofreading on diverse slices and regions.

## 11. Dataset maintenance
This repo does not host or curate the upstream dataset. Maintenance concerns include:
- BossDB dataset URI changes or access changes
- `intern` library changes affecting data retrieval

If the dataset becomes unavailable:
- update the dataset URI in `run_pipeline.py`
- or add a fallback local tiny sample volume for offline runs

## 12. Example command lines

Baseline run (recommended for 8 GB RAM machines):
```bash
python run_pipeline.py --preset safe8gb --save_minimal --aligned_dtype float16
