"""Check whether the GP DTM's predictive standard deviation is calibrated.

The GP is trained on at most MAX_TRAINING_CELLS occupied ground cells. Every
other occupied ground cell is independent check data. For those cells:

    z = (observed median ground Z - GP prediction) / GP sigma

If sigma is calibrated, z has a standard deviation of ~1 and ~95% of |z| < 1.96.

Settings below must match the run of create_surface_models.py being checked.
"""
from pathlib import Path

import laspy
import numpy as np

# --- Settings: match these to the run you want to check -----------------------
INPUT_LAS = Path(
    r"C:\Users\Hp\Desktop\Thesis\Recce Flight planning\code from Schneider"
    r"\HessigheimSubset\Mar19_test_GroundTruth_cropped_subsampled_1Mil.laz.las"
)
# Folder that create_surface_models.py wrote (contains surfaces.npz).
# Leave as None to pick the most recently modified *_dtm_gp folder next to this script.
RESULT_DIR = None
GROUND_CLASSES = [0, 1, 8]  # Hessigheim: low vegetation, impervious surface, soil/gravel
# -----------------------------------------------------------------------------


def find_result_dir():
    if RESULT_DIR is not None:
        return Path(RESULT_DIR)
    candidates = [
        p for p in Path(__file__).parent.glob("*_dtm_gp")
        if (p / "surfaces.npz").is_file()
    ]
    if not candidates:
        raise FileNotFoundError("No *_dtm_gp folder with surfaces.npz found; set RESULT_DIR.")
    return max(candidates, key=lambda p: (p / "surfaces.npz").stat().st_mtime)


def main():
    result_dir = find_result_dir()
    data = np.load(result_dir / "surfaces.npz", allow_pickle=True)
    if str(data["dtm_interpolation"]) != "gp":
        raise ValueError("This run used IDW for the DTM; its spread is not a GP sigma.")

    cell_size = float(data["cell_size"])
    ncols, nrows = int(data["ncols"]), int(data["nrows"])
    x_min, y_min = float(data["x_min"]), float(data["y_min"])
    dtm = data["dtm"].ravel()
    sigma = data["dtm_uncertainty"].ravel()
    n_train_used = int(data["dtm_training_cells"])

    # Recompute observed per-cell median ground Z exactly as the main script does.
    cloud = laspy.read(INPUT_LAS)
    x = np.asarray(cloud.x, dtype=np.float64)
    y = np.asarray(cloud.y, dtype=np.float64)
    z = np.asarray(cloud.z, dtype=np.float64)
    if not (np.isclose(x.min(), x_min) and np.isclose(y.min(), y_min)):
        raise ValueError("Grid origin does not match: wrong INPUT_LAS for this result folder?")
    cols = np.clip(np.floor((x - x_min) / cell_size).astype(np.int64), 0, ncols - 1)
    rows = np.clip(np.floor((y - y_min) / cell_size).astype(np.int64), 0, nrows - 1)
    flat = rows * ncols + cols

    ground = np.isin(np.asarray(cloud.classification), GROUND_CLASSES)
    g_idx, g_z = flat[ground], z[ground]
    order = np.argsort(g_idx, kind="stable")
    g_idx, g_z = g_idx[order], g_z[order]
    occupied, starts, counts = np.unique(g_idx, return_index=True, return_counts=True)
    observed = np.array([np.median(g_z[s:s + c]) for s, c in zip(starts, counts)])

    # Reconstruct which cells the GP was trained on (same deterministic rule).
    n_occ = len(occupied)
    if n_train_used < n_occ:
        train_pos = np.linspace(0, n_occ - 1, n_train_used, dtype=int)
    else:
        train_pos = np.arange(n_occ)
    held_out = np.ones(n_occ, dtype=bool)
    held_out[train_pos] = False

    cells = occupied[held_out]
    resid = observed[held_out] - dtm[cells]
    s = sigma[cells]
    zscore = resid / s

    print(f"Result folder: {result_dir.name}")
    print(f"Occupied ground cells: {n_occ:,}; trained on {len(train_pos):,}; "
          f"checked on {held_out.sum():,} held-out cells")
    if held_out.sum() == 0:
        print("No held-out cells: the GP used every occupied cell. Nothing to check.")
        return

    print("\nResiduals (observed - predicted), in Z units:")
    print(f"  mean {resid.mean():+.3f}   RMSE {np.sqrt(np.mean(resid**2)):.3f}   "
          f"max |r| {np.abs(resid).max():.3f}")
    print("Predicted sigma at those cells:")
    print(f"  min {s.min():.3f}   median {np.median(s):.3f}   max {s.max():.3f}")
    print("\nCalibration (expect std ~1.0, ~68% within 1, ~95% within 1.96, ~99.7% within 3):")
    print(f"  std(z) {zscore.std():.2f}")
    for k, expected in [(1.0, 68.3), (1.96, 95.0), (3.0, 99.7)]:
        frac = 100 * np.mean(np.abs(zscore) < k)
        print(f"  |z| < {k:<4}: {frac:5.1f}%   (expected {expected}%)")

    # One-sided view: how often is the true ground ABOVE the prediction by more than k*sigma?
    # This is the unsafe direction for a minimum safe-altitude surface.
    print("\nUnsafe side (ground higher than prediction + k*sigma):")
    for k, expected in [(1.645, 5.0), (2.326, 1.0)]:
        frac = 100 * np.mean(zscore > k)
        print(f"  z > {k}: {frac:5.2f}%   (expected {expected}%)")


if __name__ == "__main__":
    main()
