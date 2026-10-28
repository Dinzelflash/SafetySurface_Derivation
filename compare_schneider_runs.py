"""Part B: inspect Prof. Schneider's result folders and compare them with the
hand-labelled Hessigheim ground (classes 0, 1, 8).

Run from VS Code (F5). Prints, for every result folder in HESSIGHEIM_DIR:
  1. the settings stored in surfaces.npz (incl. how ground was classified)
  2. the sigma range
  3. accuracy of the DTM against the hand-labelled ground medians
  4. which object classes SMRF wrongly kept as ground
"""
from pathlib import Path

import laspy
import numpy as np

HESSIGHEIM_DIR = Path(
    r"C:\Users\Hp\Desktop\Thesis\Recce Flight planning\code from Schneider\HessigheimSubset"
)
INPUT_LAS = HESSIGHEIM_DIR / "Mar19_test_GroundTruth_cropped_subsampled_1Mil.laz.las"
HAND_GROUND = [0, 1, 8]  # H3D: low vegetation, impervious surface, soil/gravel
CLASS_NAMES = {0: "low vegetation", 1: "impervious", 2: "vehicle", 3: "urban furniture",
               4: "roof", 5: "facade", 6: "shrub", 7: "tree", 8: "soil/gravel",
               9: "vertical surface", 10: "chimney"}
SETTINGS = ["covariance_function", "dtm_interpolation", "dtm_training_cells", "cell_size",
            "ground_classification_mode", "ground_classification_cell_size",
            "ground_classification_window_size", "ground_classification_threshold",
            "ground_point_count", "ground_point_percentage"]


def main():
    cloud = laspy.read(INPUT_LAS)
    x, y, z = (np.asarray(v, dtype=np.float64) for v in (cloud.x, cloud.y, cloud.z))
    cls = np.asarray(cloud.classification)
    hand = np.isin(cls, HAND_GROUND)
    print(f"Input: {len(z):,} points, hand-labelled ground {hand.sum():,} ({100 * hand.mean():.1f}%)")

    folders = sorted(p for p in HESSIGHEIM_DIR.iterdir() if (p / "surfaces.npz").is_file())
    for folder in folders:
        d = np.load(folder / "surfaces.npz", allow_pickle=True)
        cs, nc, nr = float(d["cell_size"]), int(d["ncols"]), int(d["nrows"])
        x0, y0 = float(d["x_min"]), float(d["y_min"])
        col = np.clip(np.floor((x - x0) / cs).astype(int), 0, nc - 1)
        row = np.clip(np.floor((y - y0) / cs).astype(int), 0, nr - 1)
        flat = row * nc + col

        # Median hand-labelled ground height per cell (the "truth" here).
        gi, gz = flat[hand], z[hand]
        order = np.argsort(gi, kind="stable")
        gi, gz = gi[order], gz[order]
        occ, starts, counts = np.unique(gi, return_index=True, return_counts=True)
        truth = np.array([np.median(gz[s:s + n]) for s, n in zip(starts, counts)])

        dtm = d["dtm"].ravel()
        sigma = d["dtm_uncertainty"].ravel()
        smrf_occ = d["dtm_point_count"].ravel() > 0
        resid = truth - dtm[occ]
        zs = resid / sigma[occ]

        print(f"\n===== {folder.name}")
        for key in SETTINGS:
            if key in d.files:
                print(f"  {key}: {d[key]}")
        print(f"  sigma: min {sigma.min():.3f}  median {np.median(sigma):.3f}  max {sigma.max():.3f} m")
        print(f"  sigma median at cells with ground data {np.median(sigma[smrf_occ]):.3f} m, "
              f"at empty cells {np.median(sigma[~smrf_occ]):.3f} m")
        print(f"  DTM vs hand ground ({len(occ):,} cells): mean {resid.mean():+.3f}  "
              f"RMSE {np.sqrt(np.mean(resid ** 2)):.3f}  max |error| {np.abs(resid).max():.3f} m")
        print(f"  within 1 / 1.96 / 3 sigma: {100 * np.mean(abs(zs) < 1):.1f} / "
              f"{100 * np.mean(abs(zs) < 1.96):.1f} / {100 * np.mean(abs(zs) < 3):.1f} %  "
              "(expected 68.3 / 95.0 / 99.7)")
        print(f"  unsafe (ground above prediction + 2.326 sigma): {100 * np.mean(zs > 2.326):.2f} % "
              "(expected 1.00)")

        handocc = np.zeros(nc * nr, bool)
        handocc[occ] = True
        wrong = smrf_occ & ~handocc
        in_wrong = np.isin(flat, np.flatnonzero(wrong))
        u, n = np.unique(cls[in_wrong], return_counts=True)
        top = sorted(zip(n, u), reverse=True)[:4]
        print(f"  cells SMRF called ground but hand labels say no ground: {wrong.sum():,} "
              f"({100 * wrong.sum() / smrf_occ.sum():.1f}% of SMRF ground cells)")
        print("  objects in those cells: " +
              ", ".join(f"{CLASS_NAMES.get(int(k), k)} {100 * v / n.sum():.0f}%" for v, k in top))


if __name__ == "__main__":
    main()
