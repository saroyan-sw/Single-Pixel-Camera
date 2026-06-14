"""
Project 6 -- Full parametric study (tuned).

    PART A.  Reconstruction comparison at M = 1..4 with N=2000, low noise.
    PART B.  RMSE versus number of measurements, for each M.
    PART C.  RMSE versus measurement-noise level, for each M.
"""
import sys, time, os, json
sys.path.insert(0, '/home/claude')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spc_recon import (dmd_random_patterns, build_sensing_matrix,
                       reconstruct_tikhonov, reconstruct_nesta_tv,
                       rmse, resize_image)

OUT = '/home/claude/project6_out'
os.makedirs(OUT, exist_ok=True)

# 96 x 96 ground truth (divisible by 1, 2, 3, 4).
H, W = 96, 96
X_true = np.zeros((H, W), dtype=np.float32)
yy, xx = np.mgrid[:H, :W]
X_true[(yy - 28)**2 + (xx - 28)**2 < 14**2] = 1.0
X_true[55:80, 12:38] = 0.6
for i in range(22):
    X_true[58 + i, 55 + i:80 - i//2] = 0.85
X_true[10:18, 60:85] = 0.4

LAM_BY_M = {1: 500.0, 2: 100.0, 3: 10.0, 4: 1.0}

N_MAX = 4000
print("Generating patterns...", flush=True)
patterns = dmd_random_patterns(W, H, num_patterns=N_MAX, seed=42)
print(f"  {patterns.shape}  ({patterns.nbytes/1e6:.1f} MB)", flush=True)

_phi_cache = {}
def get_phi(M, N):
    if M not in _phi_cache:
        _phi_cache[M] = build_sensing_matrix(patterns, macropixel=M)
    return _phi_cache[M][:N]

def gt_at_scale(M):
    return resize_image(X_true, (H // M, W // M))

def do_tikhonov(Phi, y, shape):
    damp = float(y.std()) / np.sqrt(shape[0] * shape[1])
    return reconstruct_tikhonov(Phi, y, damp=damp, iter_lim=400).reshape(shape)

def do_nesta(Phi, y, shape, M, n_iter=150, tv_inner=12):
    return reconstruct_nesta_tv(Phi, y.astype(np.float32), image_shape=shape,
                                 lam=LAM_BY_M[M], n_iter=n_iter,
                                 tv_inner=tv_inner).reshape(shape)

rng = np.random.default_rng(0)
colors = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red'}

# -------------------- PART A --------------------
print("\n=== PART A ===", flush=True)
N_A = 2000
recon = {}
for M in [1, 2, 3, 4]:
    h, w = H // M, W // M
    Phi = get_phi(M, N_A)
    X_gt = gt_at_scale(M)
    y_clean = Phi @ X_gt.ravel()
    y = y_clean + 0.01 * y_clean.std() * rng.standard_normal(N_A)
    t0 = time.time(); x_tik = do_tikhonov(Phi, y, (h, w)); t_tik = time.time() - t0
    t0 = time.time(); x_tv = do_nesta(Phi, y, (h, w), M, n_iter=200, tv_inner=15); t_tv = time.time() - t0
    e_tik = rmse(x_tik, X_gt); e_tv = rmse(x_tv, X_gt)
    recon[M] = dict(gt=X_gt, tik=x_tik, tv=x_tv, rmse_tik=e_tik, rmse_tv=e_tv,
                    n_meas=N_A, n_unk=h*w)
    print(f"  M={M}: Tik {e_tik:.4f} ({t_tik:.1f}s)  NESTA {e_tv:.4f} ({t_tv:.1f}s)", flush=True)

fig, axes = plt.subplots(3, 4, figsize=(13, 9.5))
for i, M in enumerate([1, 2, 3, 4]):
    h, w = H // M, W // M
    axes[0, i].imshow(recon[M]['gt'], cmap='gray', vmin=0, vmax=1)
    axes[0, i].set_title(f"Ground truth\nM={M} ({h}×{w}, n={h*w})", fontsize=10)
    axes[1, i].imshow(recon[M]['tik'], cmap='gray')
    axes[1, i].set_title(f"Standard / Tikhonov\nRMSE={recon[M]['rmse_tik']:.3f}", fontsize=10)
    axes[2, i].imshow(recon[M]['tv'], cmap='gray')
    axes[2, i].set_title(f"NESTA-style (TV)\nRMSE={recon[M]['rmse_tv']:.3f}", fontsize=10)
    for ax in axes[:, i]: ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Reconstructions at N={N_A} measurements, 1 % noise", fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/figA_reconstructions.png', dpi=120, bbox_inches='tight')
np.savez(f'{OUT}/recon_partA.npz',
         **{f'M{M}_gt':  recon[M]['gt']  for M in [1,2,3,4]},
         **{f'M{M}_tik': recon[M]['tik'] for M in [1,2,3,4]},
         **{f'M{M}_tv':  recon[M]['tv']  for M in [1,2,3,4]})
print("  -> figA_reconstructions.png", flush=True)

# -------------------- PART B --------------------
print("\n=== PART B ===", flush=True)
N_grid = [200, 500, 1000, 2000, 3000, 4000]
curves = {M: {'N': [], 'tik': [], 'tv': []} for M in [1, 2, 3, 4]}
for M in [1, 2, 3, 4]:
    h, w = H // M, W // M
    X_gt = gt_at_scale(M)
    for Nm in N_grid:
        Phi = get_phi(M, Nm)
        y_clean = Phi @ X_gt.ravel()
        y = y_clean + 0.01 * y_clean.std() * rng.standard_normal(Nm)
        n_it = 80 if M == 1 else 100
        x_tik = do_tikhonov(Phi, y, (h, w))
        x_tv = do_nesta(Phi, y, (h, w), M, n_iter=n_it, tv_inner=10)
        e_tik = rmse(x_tik, X_gt); e_tv = rmse(x_tv, X_gt)
        curves[M]['N'].append(Nm); curves[M]['tik'].append(e_tik); curves[M]['tv'].append(e_tv)
        print(f"  M={M} N={Nm:4d}: Tik={e_tik:.3f} TV={e_tv:.3f}", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for M in [1, 2, 3, 4]:
    axes[0].plot(curves[M]['N'], curves[M]['tik'], '-o', color=colors[M], label=f'M={M}')
    axes[1].plot(curves[M]['N'], curves[M]['tv'],  '-o', color=colors[M], label=f'M={M}')
for ax, title in zip(axes, ['Standard (Tikhonov)', 'NESTA-style (TV)']):
    ax.set_xlabel('number of measurements N'); ax.set_ylabel('RMSE (normalised)')
    ax.set_title(title); ax.set_xscale('log'); ax.grid(True, alpha=0.4); ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/figB_rmse_vs_N.png', dpi=120, bbox_inches='tight')
print("  -> figB_rmse_vs_N.png", flush=True)

# -------------------- PART C --------------------
print("\n=== PART C ===", flush=True)
noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
N_C = 2000
ncurves = {M: {'noise': [], 'tik': [], 'tv': []} for M in [1, 2, 3, 4]}
for M in [1, 2, 3, 4]:
    h, w = H // M, W // M
    Phi = get_phi(M, N_C)
    X_gt = gt_at_scale(M)
    y_clean = Phi @ X_gt.ravel()
    for sigma in noise_levels:
        y = y_clean + sigma * y_clean.std() * rng.standard_normal(N_C)
        n_it = 80 if M == 1 else 100
        x_tik = do_tikhonov(Phi, y, (h, w))
        x_tv = do_nesta(Phi, y, (h, w), M, n_iter=n_it, tv_inner=10)
        e_tik = rmse(x_tik, X_gt); e_tv = rmse(x_tv, X_gt)
        ncurves[M]['noise'].append(sigma); ncurves[M]['tik'].append(e_tik); ncurves[M]['tv'].append(e_tv)
    print(f"  M={M}: Tik {['%.3f'%v for v in ncurves[M]['tik']]}", flush=True)
    print(f"  M={M}: TV  {['%.3f'%v for v in ncurves[M]['tv']]}", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for M in [1, 2, 3, 4]:
    axes[0].plot(ncurves[M]['noise'], ncurves[M]['tik'], '-o', color=colors[M], label=f'M={M}')
    axes[1].plot(ncurves[M]['noise'], ncurves[M]['tv'],  '-o', color=colors[M], label=f'M={M}')
for ax, title in zip(axes, ['Standard (Tikhonov)', 'NESTA-style (TV)']):
    ax.set_xlabel('relative noise level σ / std(y_clean)')
    ax.set_ylabel('RMSE (normalised)')
    ax.set_title(title); ax.grid(True, alpha=0.4); ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/figC_noise_robustness.png', dpi=120, bbox_inches='tight')
print("  -> figC_noise_robustness.png", flush=True)

results = {
    'config': {'image_shape': [H, W], 'N_max': N_MAX, 'seed': 42, 'lam_by_M': LAM_BY_M},
    'partA': {str(M): {'rmse_tik': float(recon[M]['rmse_tik']),
                       'rmse_tv':  float(recon[M]['rmse_tv']),
                       'n_unknowns': int(recon[M]['n_unk']),
                       'n_meas': int(recon[M]['n_meas'])} for M in [1,2,3,4]},
    'partB': {str(M): {'N': curves[M]['N'],
                       'rmse_tik': [float(v) for v in curves[M]['tik']],
                       'rmse_tv':  [float(v) for v in curves[M]['tv']]} for M in [1,2,3,4]},
    'partC': {str(M): {'noise': ncurves[M]['noise'],
                       'rmse_tik': [float(v) for v in ncurves[M]['tik']],
                       'rmse_tv':  [float(v) for v in ncurves[M]['tv']]} for M in [1,2,3,4]},
}
with open(f'{OUT}/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nAll outputs in {OUT}/")
for fn in sorted(os.listdir(OUT)):
    print(f"  {fn:35s} {os.path.getsize(f'{OUT}/{fn}')/1024:.1f} kB")
