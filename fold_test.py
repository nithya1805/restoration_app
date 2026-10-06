"""Fold / warp scores from UVDoc's 3D page surface, for every image in INPUT_FOLDER.

UVDoc predicts, for a 45 x 31 grid over the page, where each point sits in 3D (x, y, z).
From that surface:
  plane_rms   how far the page is from a flat plane         -> overall bending (curl, warp)
  curv_*      bending at each grid point (second difference) -> where the page bends
  peak_ratio  curv_p99 / curv_median                          -> sharp crease (high) vs smooth curl (low)
  warp_2d     the score app.py uses today (2D grid vs best affine fit)
A picture per image (original, height off the plane, curvature map) is saved to OUTPUT_FOLDER.

Run with the venv:  .venv\\Scripts\\python.exe fold_test.py
"""
import csv
import importlib.util
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")                     # save files only, no window
import matplotlib.pyplot as plt
import numpy as np
import torch

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FOLDER = r"C:\restoration_app\folded"
OUTPUT_FOLDER = r"C:\restoration_app\folded_3d_test"
CSV_PATH = os.path.join(OUTPUT_FOLDER, "fold_scores.csv")    # one row per image, with the verdict

UVDOC_CODE_DIR = Path(r"C:\UVDoc")                        # same as CONFIG in app.py
UVDOC_MODEL_PATH = Path(r"C:\UVDoc\model\best_model.pkl")
UVDOC_IMG_SIZE = (488, 712)                               # (w, h) network input

# verdict rules
WARP_THRESHOLD = 0.02         # warp_2d >= this -> WARPED (the same rule as dewarp_threshold in app.py)
PLANE_MILD = 0.02             # plane_rms below this: surface is flat
PLANE_STRONG = 0.05           # plane_rms at or above this: strong bending

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# UVDoc (loaded on its own - importing app.py would load every model)
# ============================================================

def load_uvdoc():
    spec = importlib.util.spec_from_file_location("uvdoc_model", UVDOC_CODE_DIR / "model.py")
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    model = arch.UVDocnet(num_filter=32, kernel_size=5)
    model.load_state_dict(torch.load(UVDOC_MODEL_PATH, map_location="cpu", weights_only=False)["model_state"])
    return model.eval().to(DEVICE)


UVDOC = load_uvdoc()


@torch.no_grad()
def run_uvdoc(image):
    """BGR image -> (points_2d (2, 45, 31), points_3d (3, 45, 31)) as numpy arrays."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = torch.from_numpy(cv2.resize(rgb, UVDOC_IMG_SIZE).transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    points_2d, points_3d = UVDOC(inp)
    return points_2d[0].cpu().numpy(), points_3d[0].cpu().numpy()


# ============================================================
# SCORES
# ============================================================

def warp_score_2d(points_2d):
    """The score app.py uses (deformation_score): 2D grid vs its best affine fit."""
    g = points_2d.transpose(1, 2, 0).reshape(-1, 2).astype(np.float64)
    h, w = points_2d.shape[1:]
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    flat = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=1)
    coef, *_ = np.linalg.lstsq(flat, g, rcond=None)
    return float(np.linalg.norm(g - flat @ coef, axis=1).mean())


def fold_scores(points_3d):
    """Scores from the 3D surface, scaled by the page size so they do not depend on its units."""
    p = points_3d.transpose(1, 2, 0).astype(np.float64)   # (45, 31, 3): x, y, z per grid point
    h, w, _ = p.shape
    pts = p.reshape(-1, 3)

    # best-fit plane through the points; the signed distance from it is the page's "height"
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    normal = vt[2]
    size = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))  # page diagonal
    height = ((pts - centre) @ normal).reshape(h, w) / size

    # bending at each grid point: second difference of the 3D position along rows and columns
    # (0 on a flat or evenly tilted page, large where the surface bends sharply)
    d2_row = np.linalg.norm(p[:, :-2] - 2 * p[:, 1:-1] + p[:, 2:], axis=2)   # (h, w-2)
    d2_col = np.linalg.norm(p[:-2] - 2 * p[1:-1] + p[2:], axis=2)            # (h-2, w)
    curv = np.zeros((h, w))
    curv[:, 1:-1] = np.maximum(curv[:, 1:-1], d2_row)
    curv[1:-1, :] = np.maximum(curv[1:-1, :], d2_col)
    curv /= size
    inner = curv[1:-1, 1:-1].ravel()                       # borders have only one direction

    median = float(np.median(inner))
    p99 = float(np.percentile(inner, 99))
    return {
        "plane_rms": float(np.sqrt(np.mean(height ** 2))),
        "curv_mean": float(inner.mean()),
        "curv_p99": p99,
        "curv_max": float(inner.max()),
        "peak_ratio": p99 / median if median > 0 else float("inf"),
    }, height, curv


def verdict(scores):
    """WARPED / FLAT by the app's rule, and how strongly the 3D surface is bent."""
    warped = scores["warp_2d"] >= WARP_THRESHOLD
    rms = scores["plane_rms"]
    level = "flat" if rms < PLANE_MILD else "mild" if rms < PLANE_STRONG else "strong"
    return ("WARPED" if warped else "FLAT"), level


# ============================================================
# VISUALIZATION
# ============================================================

def save_visualization(image, height, curv, scores, filename):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result, level = verdict(scores)
    ax[0].set_title(f"{result} - {level} bending (warp_2d {scores['warp_2d']:.4f})")
    im = ax[1].imshow(height, cmap="coolwarm")
    ax[1].set_title(f"Height off plane (rms {scores['plane_rms']:.4f})")
    fig.colorbar(im, ax=ax[1])
    im = ax[2].imshow(curv, cmap="hot")
    ax[2].set_title(f"Curvature (p99 {scores['curv_p99']:.4f}, peak x{scores['peak_ratio']:.1f})")
    fig.colorbar(im, ax=ax[2])
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PROCESS FOLDER
# ============================================================

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
rows = []

for filename in sorted(os.listdir(INPUT_FOLDER)):
    if not filename.lower().endswith(image_extensions):
        continue
    image = cv2.imread(os.path.join(INPUT_FOLDER, filename))
    if image is None:
        print(f"Could not read: {filename}")
        continue
    try:
        points_2d, points_3d = run_uvdoc(image)
        scores, height, curv = fold_scores(points_3d)
        scores["warp_2d"] = warp_score_2d(points_2d)
        save_visualization(image, height, curv, scores, filename.rsplit(".", 1)[0] + "_3d_analysis.png")
        rows.append((filename, scores))
    except Exception as e:
        print(f"ERROR processing {filename}: {e}")

print(f"\n{'image':<22}{'warp_2d':>9}{'plane_rms':>11}{'curv_mean':>11}{'curv_p99':>10}{'curv_max':>10}"
      f"{'peak_ratio':>12}   {'result':<8}{'bending'}")
for filename, s in rows:
    result, level = verdict(s)
    print(f"{filename:<22}{s['warp_2d']:>9.4f}{s['plane_rms']:>11.4f}{s['curv_mean']:>11.5f}"
          f"{s['curv_p99']:>10.5f}{s['curv_max']:>10.5f}{s['peak_ratio']:>12.1f}   {result:<8}{level}")

warped = [f for f, s in rows if verdict(s)[0] == "WARPED"]
print(f"\nWARPED ({len(warped)}): {', '.join(warped) or '-'}")
print(f"FLAT   ({len(rows) - len(warped)}): {', '.join(f for f, s in rows if f not in warped) or '-'}")
print(f"\nRules: WARPED if warp_2d >= {WARP_THRESHOLD} (as in app.py). "
      f"Bending from plane_rms: flat < {PLANE_MILD} <= mild < {PLANE_STRONG} <= strong.")
print("Folds are not judged: none of these scores separates folded pages reliably.")

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "result", "bending", "warp_2d", "plane_rms", "curv_mean", "curv_p99",
                     "curv_max", "peak_ratio", "picture"])
    for filename, s in rows:
        result, level = verdict(s)
        writer.writerow([filename, result, level, f"{s['warp_2d']:.4f}", f"{s['plane_rms']:.4f}",
                         f"{s['curv_mean']:.5f}", f"{s['curv_p99']:.5f}", f"{s['curv_max']:.5f}",
                         f"{s['peak_ratio']:.1f}", filename.rsplit(".", 1)[0] + "_3d_analysis.png"])

print(f"\nPictures saved in {OUTPUT_FOLDER}")
print(f"CSV saved as      {CSV_PATH}")
