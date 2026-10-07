"""
Mendeley paired thermal-electrical physical validation used by LMEF.

Expected per experiment:
  Panel{1..3}_{Clean|Dirt|Shadow}.csv  -> 80 x 61 raw radiometric CSV
  Panel{1..3}_{Clean|Dirt|Shadow}.mat  -> Vpanel, Ipanel, IRR, T (+ optional IRRs, Ts)

Important:
- The first CSV column is a spatial row coordinate (1..80), not temperature.
- After removing it, the radiometric matrix is 80 x 60.
- I-V acquisition is reconstructed with 250 voltage bins using median V/I.
- Isc_est and Voc_est are extrapolated estimates, not directly measured endpoints.
- Fixed rectangular ROIs are defined from Clean geometry and reused within panel.
- This script is physical validation, not a classifier.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROIS = {
    1: {"xmin": 10, "xmax": 55, "ymin": 6,  "ymax": 74},
    2: {"xmin": 10, "xmax": 47, "ymin": 6,  "ymax": 71},
    3: {"xmin": 9,  "xmax": 46, "ymin": 11, "ymax": 68},
}

CONDITIONS = ("Clean", "Dirt", "Shadow")

def load_radiometric_csv(path):
    raw = pd.read_csv(path, header=None).apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
    a = raw.to_numpy(dtype=float)
    # Locate an 80 x 61 block if metadata rows/cols exist.
    if a.shape != (80, 61):
        found = None
        for r in range(max(1, a.shape[0] - 79)):
            for c in range(max(1, a.shape[1] - 60)):
                b = a[r:r+80, c:c+61]
                if b.shape == (80, 61) and np.allclose(b[:,0], np.arange(1,81), equal_nan=False):
                    found = b
                    break
            if found is not None:
                break
        if found is None:
            raise ValueError(f"Could not identify 80x61 radiometric block in {path}; shape={a.shape}")
        a = found
    if not np.allclose(a[:,0], np.arange(1,81)):
        raise ValueError(f"First column is not 1..80 coordinate in {path}")
    return a[:,1:]  # 80 x 60 temperatures

def thermal_features(matrix, roi):
    x0,x1,y0,y1 = roi["xmin"],roi["xmax"],roi["ymin"],roi["ymax"]
    r = matrix[y0:y1+1, x0:x1+1]
    med = float(np.nanmedian(r))
    return {
        "ROI_mean_C": float(np.nanmean(r)),
        "ROI_median_C": med,
        "ROI_P95_C": float(np.nanpercentile(r,95)),
        "ROI_Tmax_C": float(np.nanmax(r)),
        "ROI_heterogeneity_C": float(np.nanmax(r)-med),
        "hotspot_area_median_plus_5_pct": float(100*np.mean(r > med+5.0)),
    }

def scalar(mat, key):
    if key not in mat:
        return np.nan
    return float(np.asarray(mat[key]).squeeze())

def reconstruct_iv(mat, n_bins=250):
    V = np.asarray(mat["Vpanel"]).reshape(-1).astype(float)
    I = np.asarray(mat["Ipanel"]).reshape(-1).astype(float)
    ok = np.isfinite(V) & np.isfinite(I)
    V, I = V[ok], I[ok]

    edges = np.linspace(V.min(), V.max(), n_bins+1)
    bid = np.clip(np.digitize(V, edges)-1, 0, n_bins-1)
    rows = []
    for b in range(n_bins):
        m = bid == b
        if m.sum():
            vm, im = np.median(V[m]), np.median(I[m])
            rows.append((vm, im, vm*im))
    curve = pd.DataFrame(rows, columns=["V","I","P"]).sort_values("V")
    peak = curve.loc[curve["P"].idxmax()]

    n5 = max(3, int(np.ceil(0.05*len(curve))))
    lo = curve.nsmallest(n5, "V")
    hi = curve.nlargest(n5, "V")
    # I = aV+b => Isc_est=b
    a_i,b_i = np.polyfit(lo["V"], lo["I"], 1)
    isc_est = b_i
    # V = aI+b => Voc_est=b
    a_v,b_v = np.polyfit(hi["I"], hi["V"], 1)
    voc_est = b_v
    ff = peak.P/(isc_est*voc_est) if isc_est*voc_est else np.nan

    return {
        "Vmp_V": float(peak.V), "Imp_A": float(peak.I), "Pmpp_W": float(peak.P),
        "Isc_est_A": float(isc_est), "Voc_est_V": float(voc_est), "FF_est": float(ff),
    }, curve

def analyze(root):
    root = Path(root)
    rows = []
    for panel in (1,2,3):
        for condition in CONDITIONS:
            csvp = root/f"Panel{panel}_{condition}.csv"
            matp = root/f"Panel{panel}_{condition}.mat"
            if not (csvp.exists() and matp.exists()):
                print("SKIP missing:", csvp.name, matp.name)
                continue
            Tm = load_radiometric_csv(csvp)
            mat = loadmat(matp)
            row = {"Panel":panel, "Condition":condition}
            row.update({"IRR_Wm2":scalar(mat,"IRR"), "Tpanel_C":scalar(mat,"T")})
            row.update(reconstruct_iv(mat)[0])
            row.update(thermal_features(Tm, ROIS[panel]))
            row["Pnorm"] = row["Pmpp_W"]/(row["IRR_Wm2"]/1000.0)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv("mendeley_multimodal_features.csv", index=False)
    return out

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("root", help="Folder containing Panel*_*.csv and Panel*_*.mat")
    args = p.parse_args()
    df = analyze(args.root)
    print(df.to_string(index=False))
