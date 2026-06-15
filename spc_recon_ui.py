"""
Single-Pixel Camera Reconstruction — interactive UI
===================================================

A point-and-click front end for the Project-6 reconstruction workflow.
Instead of editing notebook cells to change the macropixel size or the
reconstruction method, pick them from the sidebar and press **Run**.

Run with:
    streamlit run spc_recon_ui.py

Requires `spc_recon_2.py` (and your data files) to be reachable from the
folder you launch this from.
"""

from __future__ import annotations

import io
import re
import time
from math import gcd

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from spc_recon_2 import reconstruct_tikhonov, reconstruct_nesta_tv

# ---- optional extra reconstruction methods (only shown if importable) -------
EXTRA_METHODS = {}
try:
    from spc_cs_l1 import reconstruct_dct_l1  # type: ignore
    EXTRA_METHODS["DCT + L1 (compressed sensing)"] = "dct_l1"
except Exception:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_NUM = re.compile(r'np\.float\d*\((-?[\d.eE+-]+)\)|(-?\d+\.?\d*(?:[eE][-+]?\d+)?)')


def parse_measurements(text: str) -> np.ndarray:
    """Robust loader: handles 'one number per line' and 'np.float64(x)' formats."""
    vals = []
    for line in text.splitlines():
        m = _NUM.search(line.strip())
        if m:
            vals.append(float(m.group(1) or m.group(2)))
    return np.asarray(vals, dtype=np.float64)


def common_divisors(h: int, w: int) -> list[int]:
    """Macropixel sizes that divide both dimensions exactly."""
    g = gcd(h, w)
    return [d for d in range(1, g + 1) if g % d == 0]


@st.cache_data(show_spinner=False)
def build_phi(N: int, h_macro: int, w_macro: int, seed: int, on_prob: float) -> np.ndarray:
    """
    Sensing matrix at macropixel resolution, generated the *same* way the
    notebook does (seeded global RNG), but without materialising the full
    tiled DMD array — each MxM block is uniform, so one sample == the block.
    Result is byte-identical to the notebook's `patterns[:, ::M, ::M]/255`.
    """
    np.random.seed(seed)
    small = np.random.choice(
        [0, 255], size=(N, h_macro, w_macro, 1), p=[1 - on_prob, on_prob]
    ).astype(np.uint8)
    return (small[..., 0].astype(np.float32) / 255.0).reshape(N, -1)


@st.cache_data(show_spinner=False)
def phi_from_npy(file_bytes: bytes, M: int, n_rows: int) -> np.ndarray:
    """Load a cached full-resolution patterns .npy and subsample to macropixel grid."""
    arr = np.load(io.BytesIO(file_bytes))
    P = arr if arr.ndim == 3 else arr[..., 0]          # (N,H,W,1) -> (N,H,W)
    Phi = (P[:n_rows, ::M, ::M].astype(np.float32) / 255.0)
    return Phi.reshape(Phi.shape[0], -1)


def normalise01(v: np.ndarray) -> np.ndarray:
    mn, mx = float(v.min()), float(v.max())
    return (v - mn) / (mx - mn + 1e-12)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Single-Pixel Camera Reconstruction",
                   layout="wide", page_icon="🔬")

st.markdown(
    "<h2 style='margin-bottom:0'>🔬 Single-Pixel Camera Reconstruction</h2>"
    "<p style='color:#888;margin-top:4px'>Pick a macropixel size and a method, "
    "then press <b>Run</b> — no cell editing.</p>",
    unsafe_allow_html=True,
)

# ============================ SIDEBAR ======================================
with st.sidebar:
    st.header("1 · Measurements")
    up = st.file_uploader("Bucket readings (.txt)", type=["txt"])
    demo = st.checkbox("Use synthetic demo scene instead",
                       value=up is None,
                       help="No data file handy? Fabricate measurements from a "
                            "known scene so you can try the UI immediately.")

    st.header("2 · DMD geometry")
    c1, c2 = st.columns(2)
    H = c1.number_input("DMD height H", 16, 4000, 1140, step=1)
    W = c2.number_input("DMD width W", 16, 4000, 912, step=1)

    divs = common_divisors(H, W)
    default_idx = divs.index(76) if 76 in divs else len(divs) // 2
    M = st.selectbox(
        "Macropixel size  M",
        divs, index=default_idx,
        help="Only sizes that divide both H and W are listed, so the choice "
             "is always valid.",
    )
    h_macro, w_macro = H // M, W // M
    n_unknowns = h_macro * w_macro
    st.caption(f"Reconstruction grid **{h_macro} × {w_macro}**  →  "
               f"**{n_unknowns}** unknowns")

    st.header("3 · Patterns Φ")
    src = st.radio("Source", ["Generate (seeded)", "Load cached .npy"],
                   horizontal=False)
    seed = st.number_input("Seed", 0, 10_000, 42, step=1)
    on_prob = st.slider("ON probability", 0.1, 0.9, 0.5, 0.05)
    npy_file = None
    if src == "Load cached .npy":
        npy_file = st.file_uploader("patterns_*.npy (full DMD resolution)",
                                    type=["npy"], key="npy")

    st.header("4 · Method")
    method_names = ["Standard — Tikhonov (LSQR)", "NESTA-style — TV (FISTA)"]
    method_names += list(EXTRA_METHODS.keys())
    method = st.radio("Reconstruction", method_names)

    # method-specific parameters
    params = {}
    if method.startswith("Standard"):
        params["damp"] = st.slider("Damping (μ)", 0.0, 5.0, 0.1, 0.05)
        params["iter_lim"] = st.select_slider(
            "LSQR iterations", [500, 1000, 5000, 10000, 20000, 50000], value=20000)
    elif method.startswith("NESTA"):
        params["lam"] = st.slider("TV weight  λ", 0.0, 2.0, 0.1, 0.01,
                                  help="Higher = smoother. Lower = sharper/noisier.")
        params["n_iter"] = st.select_slider(
            "FISTA iterations", [50, 100, 200, 400, 800], value=200)
        params["tv_inner"] = st.slider("Chambolle inner iters", 5, 40, 10, 1)
    else:  # DCT + L1
        params["lam_frac"] = st.slider("λ fraction", 0.001, 0.2, 0.01, 0.001)
        params["n_iter"] = st.select_slider(
            "ISTA iterations", [50, 100, 200, 400], value=100)
        params["nonneg"] = st.checkbox("Non-negative", value=True)

    st.header("5 · Reference (optional)")
    ref_up = st.file_uploader("Reference image for RMSE", type=["png", "jpg",
                              "jpeg", "bmp"], key="ref")

    run = st.button("▶  Run reconstruction", type="primary",
                    use_container_width=True)

# ============================ MAIN =========================================
# ---- resolve measurements -------------------------------------------------
y = None
if demo:
    # build a known coarse scene and synthesise noiseless bucket readings
    x_true = np.zeros((h_macro, w_macro), np.float32)
    x_true[h_macro // 5: 3 * h_macro // 5, w_macro // 3: 2 * w_macro // 3] = 1.0
    Phi_demo = build_phi(4000, h_macro, w_macro, seed, on_prob)
    y = (Phi_demo @ x_true.ravel()).astype(np.float64)
elif up is not None:
    y = parse_measurements(up.getvalue().decode("utf-8", errors="ignore"))

if y is None:
    st.info("⬅ Upload a measurements file or tick **Use synthetic demo scene** "
            "to get started.")
    st.stop()

N = int(y.size)

# ---- measurement summary --------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Measurements N", f"{N:,}")
m2.metric("mean  ȳ", f"{y.mean():.3g}")
m3.metric("std", f"{y.std():.3g}")
m4.metric("N / unknowns", f"{N / n_unknowns:.1f}×")
if N < n_unknowns:
    st.warning(f"Underdetermined: {N} measurements < {n_unknowns} unknowns. "
               f"Increase M or acquire more patterns.")

with st.expander("Measurement trace"):
    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.plot(y, lw=0.5)
    ax.set_xlabel("pattern index $k$"); ax.set_ylabel("$y_k$")
    ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)

# ---- run ------------------------------------------------------------------
if not run:
    st.caption("Set your options in the sidebar, then press **Run**.")
    st.stop()

# build Phi (cached -> instant when only the method/params change)
with st.spinner("Building sensing matrix Φ …"):
    if src == "Load cached .npy":
        if npy_file is None:
            st.error("Choose a patterns .npy file, or switch to 'Generate (seeded)'.")
            st.stop()
        Phi = phi_from_npy(npy_file.getvalue(), M, N)
    else:
        Phi = build_phi(N, h_macro, w_macro, seed, on_prob)

if Phi.shape[0] != N:
    st.error(f"Φ has {Phi.shape[0]} rows but there are {N} measurements — "
             f"they must match (pattern k ↔ measurement k).")
    st.stop()

# reconstruct
t0 = time.time()
with st.spinner(f"Reconstructing — {method} …"):
    if method.startswith("Standard"):
        x = reconstruct_tikhonov(Phi, y, damp=params["damp"],
                                 iter_lim=params["iter_lim"])
    elif method.startswith("NESTA"):
        x = reconstruct_nesta_tv(Phi, y.astype(np.float32), (h_macro, w_macro),
                                 lam=params["lam"], n_iter=params["n_iter"],
                                 tv_inner=params["tv_inner"])
    else:
        x = reconstruct_dct_l1(Phi, y, (h_macro, w_macro),
                               lam_frac=params["lam_frac"],
                               n_iter=params["n_iter"], nonneg=params["nonneg"])
dt = time.time() - t0
X = np.asarray(x).reshape(h_macro, w_macro)

# ---- reference (optional) -------------------------------------------------
ref_img = None
if ref_up is not None:
    from PIL import Image
    ri = Image.open(io.BytesIO(ref_up.getvalue())).convert("L")
    ri = ri.resize((w_macro, h_macro))           # (W,H) order for PIL
    ref_img = np.asarray(ri, dtype=np.float32)
elif demo:
    ref_img = x_true

# ---- display --------------------------------------------------------------
st.subheader("Result")
r1, r2, r3 = st.columns(3)
r1.metric("Runtime", f"{dt:.2f} s")
r2.metric("Value range", f"[{X.min():.3g}, {X.max():.3g}]")
if ref_img is not None:
    rmse = float(np.sqrt(((normalise01(X) - normalise01(ref_img)) ** 2).mean()))
    r3.metric("nRMSE vs reference", f"{rmse:.4f}")

ncols = 2 if ref_img is not None else 1
fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 5))
axes = np.atleast_1d(axes)
im0 = axes[0].imshow(X, cmap="gray", interpolation="nearest", aspect="equal")
axes[0].set_title(f"{method}\nM={M}  ({h_macro}×{w_macro})")
axes[0].axis("off")
fig.colorbar(im0, ax=axes[0], fraction=0.046)
if ref_img is not None:
    im1 = axes[1].imshow(ref_img, cmap="gray", interpolation="nearest", aspect="equal")
    axes[1].set_title("Reference")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
fig.tight_layout()
st.pyplot(fig, clear_figure=True)

# downloadable reconstruction
buf = io.BytesIO()
np.save(buf, X)
st.download_button("⬇  Download reconstruction (.npy)", buf.getvalue(),
                   file_name=f"recon_M{M}_{method.split()[0].lower()}.npy")
