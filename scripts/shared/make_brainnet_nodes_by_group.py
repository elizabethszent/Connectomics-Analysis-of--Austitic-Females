# scripts/shared/make_brainnet_nodes_by_group.py

import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\eliza\CPSC_599_CONNECTOMICS\TERMProject")

# Node template (we only need coords + labels from here)
TEMPLATE_NODE = BASE / r"results\vis\brainnet\CC200_base.node"

# Group-mean connectivity matrices (one per sex × diagnosis)
GROUP_MATS = {
    "female_ASD":     BASE / r"results\group_connectomes\F_ASD_Zmean.npy",
    "female_Control": BASE / r"results\group_connectomes\F_CTL_Zmean.npy",
    "male_ASD":       BASE / r"results\group_connectomes\M_ASD_Zmean.npy",
    "male_Control":   BASE / r"results\group_connectomes\M_CTL_Zmean.npy",
}

# Module labels per ROI (1..K), computed above
MODULES_FILE = BASE / r"results\group_connectomes\CC200_modules.npy"

OUT_DIR = BASE / r"results\vis\brainnet\nodes_by_group"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD TEMPLATE NODE FILE ===
# BrainNet .node format: x y z color size label
# We only trust coords + labels from this file
template = np.loadtxt(TEMPLATE_NODE, dtype=str)

if template.shape[1] < 6:
    raise ValueError(
        f"Expected 6 columns in {TEMPLATE_NODE}, found {template.shape[1]}."
        " BrainNet .node should be: x y z color size label."
    )

coords = template[:, 0:3].astype(float)   # (200, 3)
labels = template[:, 5]                   # ROI names

n_rois = coords.shape[0]
print(f"Template has {n_rois} nodes")

# === LOAD MODULES (COLUMN 4) ===
if not MODULES_FILE.exists():
    raise FileNotFoundError(f"Missing modules file: {MODULES_FILE}")

modules = np.load(MODULES_FILE)
if modules.shape[0] != n_rois:
    raise ValueError(
        f"Modules length {modules.shape[0]} does not match node file ({n_rois})"
    )

print(f"Loaded modules with {modules.max()} distinct modules")

def compute_node_strength(mat: np.ndarray) -> np.ndarray:
    """
    Weighted degree (strength). Symmetric [n, n] matrix.
    Negative weights -> 0.
    """
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Matrix is not square: {mat.shape}")
    if mat.shape[0] != n_rois:
        raise ValueError(
            f"Matrix size {mat.shape[0]} does not match node file ({n_rois})"
        )

    M = np.array(mat, copy=True)
    M[M < 0] = 0.0
    strength = M.sum(axis=0)
    return strength

def scale_to_1_10(x: np.ndarray) -> np.ndarray:
    """
    Scale a vector to [1, 10] for node sizes.
    """
    x = x.astype(float)
    xmin, xmax = x.min(), x.max()
    if np.isclose(xmax, xmin):
        return np.ones_like(x) * 5.0
    return 1.0 + 9.0 * (x - xmin) / (xmax - xmin)

# === PROCESS EACH GROUP ===
for name, mat_path in GROUP_MATS.items():
    if not mat_path.exists():
        print(f"[WARN] Missing matrix for {name}: {mat_path}")
        continue

    print(f"\nProcessing group: {name}")
    mat = np.load(mat_path)

    # Node strength for this group
    strength = compute_node_strength(mat)
    size_col = scale_to_1_10(strength)   # COLUMN 5

    out_path = OUT_DIR / f"{name}_strength.node"

    # Write BrainNet .node: x y z (4)module (5)size (6)label
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n_rois):
            x, y, z = coords[i]
            module_id = modules[i]
            size = size_col[i]
            label = labels[i]
            f.write(f"{x:.3f}\t{y:.3f}\t{z:.3f}\t{module_id:.0f}\t{size:.3f}\t{label}\n")

    print(f"  -> Saved {out_path}")
