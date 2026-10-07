# LMEF: Lightweight Multi-Modality Evaluation Framework for PV Fault Detection

Reproducibility package accompanying the manuscript:

**A Lightweight Multi-Modality Evaluation Framework for Photovoltaic Fault Detection: Electroluminescence, UAV Thermal Imaging, DC Diagnostics, and Inverter Telemetry with Thermal–Electrical Validation**

## Scope

The repository contains the executable code currently available from the study workflow. It does **not** redistribute third-party raw datasets.

### Included

- `notebooks/PVF10_Final_Training.ipynb` — final PVF-10 training/evaluation pipeline preserving the official train/test split and deriving validation data only from the original training partition.
- `notebooks/PV_Thermal_Fault_Detection_Training.ipynb` — general real thermal-image training pipeline used during the experimental workflow.
- `scripts/pvf10_training.py` — code-cell export of the final PVF-10 notebook.
- `scripts/thermal_fault_training.py` — code-cell export of the thermal training notebook.
- `scripts/mendeley_physical_validation.py` — paired radiometric thermal + electrical I–V physical-validation workflow used for the final Mendeley analysis.

## Final locked results used in the manuscript

| Dataset / branch | Main result |
|---|---|
| ELPV | 64.1% ± 7.6% five-fold CV accuracy; 208,226 parameters |
| PVF-10 | 72.7437% held-out test accuracy; macro-F1 0.68186; 1,147,626 parameters |
| GPVS-Faults | 97.3% ± 0.7% five-fold CV accuracy; 26,213 parameters |
| La Réunion inverter | 89.74% test accuracy; 89.1% macro recall; 26,472 parameters |
| Mendeley physical validation | shading-associated normalized reconstructed power reduction 83.40–85.75% (mean 84.22%) |

The Mendeley branch is **physical validation**, not an additional classifier.

## Reproducibility rules

- Random seed used in the available PVF-10 notebook: `42`.
- PVF-10 official test partition is never used for validation or model selection.
- Radiometric CSVs are interpreted as an 80×61 raw table whose first column is the spatial row coordinate; the resulting thermal matrix is 80×60.
- Mendeley I–V characteristics are reconstructed using 250 voltage bins and median V/I per bin.
- `Isc_est` and `Voc_est` are extrapolated estimates, not directly measured endpoints.
- Fixed rectangular ROIs are selected from Clean geometry once per physical panel and reused for Dirt/Shadow.
- The 500,000 electrical acquisition points per experiment are repeated/nested measurements and must not be treated as independent classifier samples.
- No sample-level fusion is performed between independent datasets.

## Data

Obtain raw data from the original providers:

- ELPV: public ELPV dataset (ZAe Bayern / original authors).
- PVF-10: Wang et al., *Applied Energy* (2024), Article 124187.
- GPVS-Faults: Mendeley Data, DOI `10.17632/n76t439f65.1`.
- La Réunion inverter data: Zenodo DOI `10.5281/zenodo.7157424` (updated record also available from Zenodo).
- Paired thermal–electrical PV experiments: Mendeley Data DOI `10.17632/xjs42j8dtf.2`.

Dataset licenses and access conditions remain those of the original providers.

## Colab use

1. Open the desired notebook in Google Colab.
2. Mount Google Drive.
3. Set the dataset root path in the configuration cell.
4. Run cells in order.
5. Save generated JSON/CSV/PNG outputs together with the environment information for the run.

## Environment

The notebooks install their principal Python dependencies at runtime. Exact CPU/RAM/GPU and package versions should be captured from the execution session used for a reported benchmark. Do not copy latency values across different hardware.

## Important limitation

The manuscript contains additional ELPV, GPVS-Faults, inverter, ablation and Grad-CAM experiments. Their final numerical results are documented in the manuscript, but the exact original executable notebooks/scripts for every one of those experiments were not present in the current code bundle used to build this repository package. They should be added before claiming that this repository reproduces every figure in the paper.

## Citation

See `CITATION.cff`. Add the final journal DOI after publication.
