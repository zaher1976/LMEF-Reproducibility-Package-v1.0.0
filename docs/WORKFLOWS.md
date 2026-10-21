# Inputs, processing and outputs

## PVF-10 classifier

Open [PVF10_Final_Training.ipynb](../PVF10_Final_Training.ipynb). `ROOT` must contain `train` and `test`, each with class folders and images directly inside. JPG, JPEG, PNG, BMP, TIF and TIFF are accepted. Leading digits are stripped and class names lowercased: `01bottom dirt` becomes `bottom dirt`. Inspect the mapping for unintended label merges.

Processing: inspect sample modes/ranges; average multichannel arrays to grayscale; divide by 65535 when the maximum exceeds 255, otherwise by 255; quantize to 8-bit; resize to 128 × 128. This heuristic is not radiometric calibration and may misidentify low-valued 16-bit inputs. Failed loads are printed and dropped.

The supplied test split is retained. The supplied training split is divided 85%/15% for training/validation with stratification and seed 42. Flips and quarter-turns apply only to training samples. This is not five-fold CV.

`LightCNN10` uses convolution channels 16/32/64/128 with batch normalization, ReLU and pooling, followed by a 128-unit head and dropout 0.3. Training uses inverse-frequency weighted cross-entropy, Adam (0.001), batch 32, at most 40 epochs, early stopping patience 8 and learning-rate reduction on validation loss. The best validation-loss state is held in memory for evaluation.

Outputs: `pvf10_results.json` and `pvf10_confusion_matrix.png`. JSON contains class order, split counts, parameter count, accuracy, macro precision/recall/F1, confusion matrix, CPU mean/p95 latency, optional GPU mean latency, epoch count and history.

## General thermal-image classifier

Open [PV_Thermal_Fault_Detection_Training.ipynb](../PV_Thermal_Fault_Detection_Training.ipynb). Upload a ZIP with class directories at the top level. It extracts to `/content/dataset`; `DATA_ROOT` must be the direct parent of the class directories. Reads JPG, JPEG, PNG and BMP directly within them, not recursively. This enumerator does not accept TIFF.

Pillow verification and a file-size threshold **greater than 20,000 bytes** filter inputs. This can reject valid small images and does not establish image quality. Preprocessing uses grayscale conversion, autocontrast (cutoff 1), a 3 × 3 median filter, 128 × 128 resizing and division by 255. These intensity images are not calibrated temperature arrays.

Two stratified image-level splits give approximately 70% training, 15% validation and 15% test (seed 42). Small classes can cause split errors. There is no panel/site grouping or duplicate detection; related images may need a different splitting design. `LightCNN` uses channels 16/32/64/64 and a 64-unit head, unlike PVF-10. Training settings match PVF-10 except early stopping patience 7.

Outputs: `thermal_fault_results.json` and `confusion_matrix.png`. JSON adds estimated FP32 parameter size and optional GPU p95 latency to the general metrics. When a `normal` class exists, binary fault-versus-normal FNR/FPR are printed but not included in JSON. Optional pilot images are counted only for qualitative use.

## Paired thermal–electrical features

Run `python mendeley_physical_validation.py /path/to/paired_files`. Use matching `Panel{1,2,3}_{Clean,Dirt,Shadow}.csv` and `.mat` files: nine pairs in total. For example, `Panel1_Clean.csv` pairs with `Panel1_Clean.mat`. This naming convention does not identify an upstream dataset DOI.

| Input | Required contents |
|---|---|
| CSV | 80 × 61 raw radiometric block. First column is row coordinate 1…80; remaining 60 columns are temperatures. Parser attempts to locate the block when metadata surrounds it. |
| MAT | `Vpanel`, `Ipanel`: paired voltage/current samples. `IRR`: scalar W/m². `T`: scalar panel °C. Loaded with `scipy.io.loadmat`. |

The existing script skips missing pairs and can write an incomplete CSV. Use the layout checker and confirm nine output rows for a complete grid. Inspect numerical inputs for missing/nonfinite values, zero irradiance and sufficient distinct voltage/current samples; preflight does not check these.

The algorithm takes fixed panel-specific inclusive NumPy-indexed ROIs, reconstructs an I–V curve with medians within 250 voltage bins, finds the maximum V×I point, then extrapolates endpoints from low-/high-voltage subsets. ROIs are assumptions for the source geometry and need inspection for other images.

| CSV fields | Meaning |
|---|---|
| `Panel`, `Condition` | Pair identity |
| `IRR_Wm2`, `Tpanel_C` | Scalar environmental measurements |
| `Vmp_V`, `Imp_A`, `Pmpp_W` | Maximum-power point on reconstructed binned curve |
| `Isc_est_A`, `Voc_est_V`, `FF_est` | Extrapolated endpoints and derived fill factor |
| `ROI_mean_C`, `ROI_median_C`, `ROI_P95_C`, `ROI_Tmax_C` | ROI temperature statistics |
| `ROI_heterogeneity_C` | Maximum minus median temperature |
| `hotspot_area_median_plus_5_pct` | ROI percentage above its median + 5 °C |
| `Pnorm` | `Pmpp_W / (IRR_Wm2 / 1000)`: irradiance-normalized power, not efficiency |

`mendeley_multimodal_features.csv` is written to the current working directory. This workflow extracts features and produces no classifier accuracy.

## Troubleshooting and run records

| Symptom | Action |
|---|---|
| `No module named google.colab` | Use the training notebook in Colab, rather than the Python export locally. |
| PVF-10 path missing | Mount Drive, edit `ROOT`, confirm train/test directories. |
| Zero general-workflow images | Check nesting, supported extensions and the byte-size filter. |
| Stratified split fails | Check per-class counts after filtering; obtain adequate samples. |
| Missing `seaborn` | Install requirements or rerun the notebook setup cell. |
| Mendeley prints `SKIP missing` | Complete the nine pairs; filenames are case-sensitive on Linux. |
| Colab files disappear | Download outputs before runtime reset; trained weights are only in memory. |

Keep data citation/version/licence, file hashes or listing, class mapping, split membership, filtering counts, code commit, installed versions, hardware, console logs and generated files. Current notebooks do not export all of these automatically. Folder enumeration is not fully sorted, dependencies are unpinned, and seed 42 alone does not ensure identical runs. Timing uses synthetic input after warmup and excludes image I/O and preprocessing.
