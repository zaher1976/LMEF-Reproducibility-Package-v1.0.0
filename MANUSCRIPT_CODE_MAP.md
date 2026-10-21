# Manuscript-to-code map

| Manuscript component | Code in this package | Status |
|---|---|---|
| PVF-10 training/test evaluation | [PVF10_Final_Training.ipynb](PVF10_Final_Training.ipynb) | Code included; supplied test split, 15% training holdout for validation |
| General thermal CNN workflow | [PV_Thermal_Fault_Detection_Training.ipynb](PV_Thermal_Fault_Detection_Training.ipynb) | Code included; image-level 70%/15%/15% split |
| Mendeley thermal–electrical validation | [mendeley_physical_validation.py](mendeley_physical_validation.py) | Feature-extraction code included; not classification |
| ELPV primary 5-fold CV | — | Original executable file not present in current bundle |
| ELPV ablation | — | Original executable file not present in current bundle |
| ELPV Grad-CAM | — | Original executable file not present in current bundle |
| GPVS-Faults 5-fold CV | — | Original executable file not present in current bundle |
| La Réunion classifier | — | Original executable file not present in current bundle |

This table is intentionally explicit so the repository does not overstate reproducibility.

All linked files are in the repository root. The Python training exports retain Colab operations; use notebooks for training. Datasets, original checkpoints, split manifests and measured run artifacts are not bundled. The package does not identify exact upstream dataset versions or DOIs. Establish data provenance and protocol correspondence before comparing a new run with a manuscript table or figure. See [workflow details](docs/WORKFLOWS.md).

