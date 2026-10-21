# LMEF — Photovoltaic Fault Detection

Lightweight Multi-Modality Evaluation Framework: research workflows for thermal-image classification and paired thermal–electrical analysis of photovoltaic panels.

**ابدأ هنا:** هذه حزمة بحثية فيها ثلاثة مسارات مستقلة، وليست تطبيقاً بواجهة استخدام. اختر المسار حسب بياناتك من الجدول. التدريب يعمل في Colab، وتحليل القياسات يعمل محلياً. البيانات والأوزان المدربة ونتائج التجارب الأصلية غير مرفقة.

[الدليل العربي](docs/START_HERE_AR.md) · [Input and output reference](docs/WORKFLOWS.md) · [Manuscript coverage](MANUSCRIPT_CODE_MAP.md)

## Choose your starting point

| Goal | Entry point | Input | Output |
|---|---|---|---|
| PVF-10 classification with its supplied test split | [PVF10_Final_Training.ipynb](PVF10_Final_Training.ipynb) | Drive directory with `train/<class>/images` and `test/<class>/images` | `pvf10_results.json`, `pvf10_confusion_matrix.png` |
| General thermal-image classification | [PV_Thermal_Fault_Detection_Training.ipynb](PV_Thermal_Fault_Detection_Training.ipynb) | ZIP containing class folders directly at its root | `thermal_fault_results.json`, `confusion_matrix.png` |
| Paired physical-feature extraction | [mendeley_physical_validation.py](mendeley_physical_validation.py) | Nine matching CSV/MAT pairs | `mendeley_multimodal_features.csv` |

The CNN workflows use different models, preprocessing and splits; choose one, rather than running one after the other. The paired analysis is not a classifier, and its features are not automatically fused with a CNN.

## Train in Google Colab

1. Obtain the relevant dataset separately and record its source, version and licence. Dataset downloads and credentials are not included.
2. Open the matching notebook and save a copy in Drive. Run cells in order and read the numbered explanations.
3. For PVF-10, mount Drive and edit `ROOT` in the data cell. For general thermal images, upload your ZIP when prompted and check `DATA_ROOT` points at the class-folder parent.
4. Inspect class names and train/validation/test counts before training. A GPU is optional; CPU training can be slower.
5. Download both generated files at the end. Use a separate folder for each run.

- [Open PVF-10 in Colab](https://colab.research.google.com/github/zaher1976/LMEF-Reproducibility-Package-v1.0.0/blob/main/PVF10_Final_Training.ipynb)
- [Open general thermal workflow in Colab](https://colab.research.google.com/github/zaher1976/LMEF-Reproducibility-Package-v1.0.0/blob/main/PV_Thermal_Fault_Detection_Training.ipynb)

The files `pvf10_training.py` and `thermal_fault_training.py` are readable notebook exports. They retain `google.colab` imports, interactive operations and Colab paths; **they are not standalone local training commands**. Use the notebooks for training.

## Check the data layout

From the repository folder, choose one command and substitute your data path. This standard-library checker does not train a model or modify data.

```bash
python check_inputs.py pvf10 --root /path/to/PVF_10_Ori
python check_inputs.py thermal --root /path/to/class_folders
python check_inputs.py mendeley --root /path/to/paired_files
```

A successful check confirms folder/file layout only. It does not validate image contents, label correctness, sample independence, numerical arrays or scientific suitability.

## Run physical analysis locally

```bash
git clone https://github.com/zaher1976/LMEF-Reproducibility-Package-v1.0.0.git
cd LMEF-Reproducibility-Package-v1.0.0
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python check_inputs.py mendeley --root /path/to/paired_files
python mendeley_physical_validation.py /path/to/paired_files
python -m pip freeze > environment-used.txt
```

The feature CSV is written to the current directory. Keep it with the source-data provenance and console log. Dependency versions are unpinned; choose a compatible environment and record its versions for each run.

## Repository guide

| File | Purpose |
|---|---|
| `docs/START_HERE_AR.md` | Arabic instructions |
| `docs/WORKFLOWS.md` | Data schemas, processing steps, result interpretation and troubleshooting |
| `check_inputs.py` | Preflight layout check |
| Two `.ipynb` notebooks | Interactive Colab training |
| Two `*_training.py` files | Readable exports of training code |
| `mendeley_physical_validation.py` | Paired physical-feature extraction |
| `MANUSCRIPT_CODE_MAP.md` | Included and missing manuscript experiments |
| `requirements.txt` | Unpinned dependencies |
| `CITATION.cff` | Package citation metadata |

## Reproducibility status

This is a **partial research-code package**. The original ELPV five-fold CV, ablation and Grad-CAM implementations, GPVS-Faults five-fold CV and La Réunion classifier are absent. The repository also has no bundled datasets, trained checkpoints, split manifests or measured result files. Consult the [coverage map](MANUSCRIPT_CODE_MAP.md) before using the package to support a manuscript result.

The notebooks use seed 42, but folder enumeration, unpinned libraries and device behaviour can change a new run. Record data identity, class mappings, split membership, filtering counts, source commit, hardware and dependency versions. The current notebooks export metrics and plots, but not trained weights or per-image predictions. Code availability alone does not establish reproduction of a published number.

Use [CITATION.cff](CITATION.cff) for package metadata. Third-party data retain their own licences. The package does not specify dataset DOIs or exact upstream versions; dataset names alone are insufficient provenance.
