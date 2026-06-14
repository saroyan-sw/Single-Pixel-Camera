"""
M = 76 macropixel reconstruction from real lab measurements.

patterns76.txt : 4000 bucket readings acquired with physically tiled
                 M=76 random binary macropixel patterns (seed=42).

Grid:  1140 // 76 = 15 rows,  912 // 76 = 12 cols  →  180 unknowns.
With 4000 measurements the system is ~22× overdetermined.
"""
import re, time
import numpy as np
import matplotlib.pyplot as plt

from spc_recon_2 import (build_phi_macropixel,
                         reconstruct_tikhonov,
                         reconstruct_nesta_tv)

# ── parameters ────────────────────────────────────────────────────────────
M    = 76
H, W = 1140, 912
h, w = H // M, W // M          # 15 × 12
SEED = 42
MEAS_FILE = 'means/patterns76.txt'

DAMP      = None    # set to None → auto (y.std / sqrt(n_unknowns))
LAM       = 5e-4    # TV regularisation weight  (small: system already overdetermined)
NESTA_ITER = 200
TV_INNER   = 15

# ── load measurements ──────────────────────────────────────────────────────
# File was saved with Python's default repr: "np.float64(2.623...)"
# Extract the number inside the parentheses.
with open(MEAS_FILE) as fh:
    y = np.array([float(ln.strip().split('(')[1].rstrip(')'))
                  for ln in fh if ln.strip()], dtype=np.float64)

N = y.size
print(f'Measurements : N={N}')
print(f'               mean={y.mean():.4f}  std={y.std():.4f}  '
      f'range=[{y.min():.4f}, {y.max():.4f}]')
print(f'Scene signal : std/mean = {y.std()/y.mean()*100:.2f}%')
print(f'Macropixel grid: {h} × {w} = {h*w} unknowns   '
      f'ratio N/n_unk = {N/(h*w):.1f}×')

# ── sensing matrix  Phi  (N × h*w) ────────────────────────────────────────
print(f'\nBuilding Phi ({N} × {h*w}) with build_phi_macropixel(seed={SEED}) ...')
t0 = time.time()
Phi = build_phi_macropixel(N, h, w, seed=SEED)
print(f'  done in {time.time()-t0:.2f}s   '
      f'shape={Phi.shape}  dtype={Phi.dtype}')

# ── Tikhonov reconstruction ────────────────────────────────────────────────
if DAMP is None:
    DAMP = float(y.std()) / float(np.sqrt(h * w))

print(f'\nTikhonov (damp={DAMP:.4g}) ...')
t0 = time.time()
X_tik = reconstruct_tikhonov(Phi, y, damp=DAMP, iter_lim=500).reshape(h, w)
print(f'  done in {time.time()-t0:.1f}s   '
      f'range=[{X_tik.min():.4g}, {X_tik.max():.4g}]')

# ── TV / NESTA reconstruction ──────────────────────────────────────────────
print(f'\nNESTA-TV (lam={LAM}, {NESTA_ITER} iter) ...')
t0 = time.time()
X_tv = reconstruct_nesta_tv(
    Phi, y.astype(np.float32),
    image_shape=(h, w),
    lam=LAM, n_iter=NESTA_ITER, tv_inner=TV_INNER,
    verbose=True
).reshape(h, w)
print(f'  done in {time.time()-t0:.1f}s   '
      f'range=[{X_tv.min():.4g}, {X_tv.max():.4g}]')

# ── visualise ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))

# measurements
axes[0].plot(y, lw=0.6, color='steelblue')
axes[0].axhline(y.mean(), color='red', ls='--', lw=0.8, label=f'mean={y.mean():.3f}')
axes[0].set_title(f'Bucket readings  (N={N})')
axes[0].set_xlabel('pattern index k')
axes[0].set_ylabel('y_k')
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# Tikhonov
im1 = axes[1].imshow(X_tik, cmap='gray', aspect='equal',
                     interpolation='nearest')
axes[1].set_title(f'Tikhonov  ({h}×{w})  damp={DAMP:.2g}')
plt.colorbar(im1, ax=axes[1], fraction=0.08)
axes[1].axis('off')

# TV
im2 = axes[2].imshow(X_tv, cmap='gray', aspect='equal',
                     interpolation='nearest')
axes[2].set_title(f'TV / NESTA  lam={LAM}')
plt.colorbar(im2, ax=axes[2], fraction=0.08)
axes[2].axis('off')

plt.suptitle(
    f'M={M} macropixel reconstruction — {h}×{w} image from {N} measurements',
    fontsize=12)
plt.tight_layout()

OUT = 'recon_m76.png'
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f'\nSaved {OUT}')
plt.show()

# ── save arrays ───────────────────────────────────────────────────────────
np.save('recon_m76_tikhonov.npy', X_tik)
np.save('recon_m76_tv.npy', X_tv)
print('Saved recon_m76_tikhonov.npy and recon_m76_tv.npy')
