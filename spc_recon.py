"""
Single-Pixel Camera reconstruction utilities.

For Project 6: Single-Pixel Camera Imaging with Variable DMD Macropixel Size.

The single-pixel camera measurement model is

    y_k = < P_k , X >   (up to an overall scale factor that we absorb into X),

where P_k is the k-th binary DMD pattern, X is the unknown scene at the DMD
plane, and y_k is the scalar bucket reading.

When the DMD is grouped into M x M macropixels, the unknown image lives on a
coarser grid of shape (H/M, W/M).  In that case each row of the sensing matrix
is the block-averaged pattern, flattened.  If X is approximately piecewise-
constant on M x M blocks, this is mathematically equivalent to running the
experiment with M x M macropixel patterns -- which is what the project asks
for when only one dataset is available.

Two reconstruction methods are provided:

    1.  "standard"  : Tikhonov-regularised least squares (LSQR with damping)
    2.  "nesta"     : TV-regularised L1 minimisation with FISTA acceleration
                      (Nesterov-style, in the spirit of NESTA)
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import lsqr, LinearOperator


# ---------------------------------------------------------------------------
# 1.  Pattern generation (matches the notebook used in the lab)
# ---------------------------------------------------------------------------

def dmd_random_patterns(width: int, height: int,
                        num_patterns: int = 2000,
                        on_probability: float = 0.5,
                        seed: int = 42,
                        dtype=np.uint8) -> np.ndarray:
    """
    Re-generate the exact random binary patterns used in the lab notebook.

    The lab notebook used::

        np.random.seed(42)
        patterns = np.random.choice([0, 255],
                                    size=(N, H, W, 1),
                                    p=[0.5, 0.5]).astype(np.uint8)

    so as long as we use the same seed and the same call to np.random.choice
    we will recover bit-identical patterns.

    Returns
    -------
    patterns : ndarray of shape (num_patterns, height, width), values {0, 255}.
    """
    np.random.seed(seed)
    patterns = np.random.choice([0, 255],
                                size=(num_patterns, height, width, 1),
                                p=[1 - on_probability, on_probability]
                                ).astype(dtype)
    # Drop the trailing singleton axis that the DMD driver expects.
    return patterns[..., 0]


def block_average(P: np.ndarray, M: int) -> np.ndarray:
    """
    Block-average each pattern over M x M micromirror blocks.

    Parameters
    ----------
    P : ndarray of shape (N, H, W) or (H, W)
        Patterns to average.  H and W must be divisible by M.
    M : int
        Macropixel size.

    Returns
    -------
    P_macro : ndarray of shape (N, H/M, W/M) or (H/M, W/M), dtype float32,
              values in [0, 255].
    """
    if M == 1:
        return P.astype(np.float32)

    if P.ndim == 2:
        H, W = P.shape
        assert H % M == 0 and W % M == 0, "H and W must be divisible by M"
        return P.reshape(H // M, M, W // M, M).mean(axis=(1, 3)).astype(np.float32)

    N, H, W = P.shape
    assert H % M == 0 and W % M == 0, "H and W must be divisible by M"
    return P.reshape(N, H // M, M, W // M, M).mean(axis=(2, 4)).astype(np.float32)


def crop_to_multiple(P: np.ndarray, M: int) -> np.ndarray:
    """
    Crop the spatial dims of P so they become divisible by M.
    """
    if P.ndim == 2:
        H, W = P.shape
        return P[:H - H % M, :W - W % M]
    N, H, W = P.shape
    return P[:, :H - H % M, :W - W % M]


# ---------------------------------------------------------------------------
# 2.  Sensing-matrix construction
# ---------------------------------------------------------------------------

def build_phi_physical(M, H, W, N, seed=42):
    """
    Correct physical simulation: generate binary patterns at macropixel
    resolution. Each M×M block on the DMD is uniformly ON or OFF.
    """
    np.random.seed(seed)
    h, w = H // M, W // M
    # Generate directly at macro resolution — this IS what the DMD displays
    small = np.random.choice([0, 1], size=(N, h, w),
                             p=[0.5, 0.5]).astype(np.float32)
    Phi = small.reshape(N, -1)   # (N, h*w)
    return Phi


def build_phi_from_existing_1x1(patterns, M):
    """
    What the current code does: re-interpret existing 1×1 measurements
    at a coarser scale via block-averaging. NOT physically equivalent to
    a true macropixel experiment, but useful for post-hoc analysis of
    an already-acquired 1×1 dataset.
    """
    P_macro = block_average(patterns, M)          # fractional values!
    return P_macro.reshape(patterns.shape[0], -1) / 255.0

# ---------------------------------------------------------------------------
# 3.  Reconstruction -- standard (Tikhonov)
# ---------------------------------------------------------------------------

def reconstruct_tikhonov(Phi: np.ndarray, y: np.ndarray,
                         damp: float = 1.0,
                         iter_lim: int = 500) -> np.ndarray:
    """
    Tikhonov-regularised least squares: argmin || Phi x - y ||_2^2 + damp^2 ||x||_2^2.
    Implemented with scipy's LSQR (damped).
    """
    y = np.asarray(y, dtype=np.float64)
    Phi = np.asarray(Phi, dtype=np.float64)
    res = lsqr(Phi, y, damp=damp, iter_lim=iter_lim, show=False)
    return res[0]


# ---------------------------------------------------------------------------
# 4.  Reconstruction -- NESTA-style TV-regularised
# ---------------------------------------------------------------------------

def _grad_2d(x):
    """Forward differences with zero boundary."""
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx[:, :-1] = x[:, 1:] - x[:, :-1]
    gy[:-1, :] = x[1:, :] - x[:-1, :]
    return gx, gy


def _div_2d(gx, gy):
    """Adjoint (negative divergence) of _grad_2d."""
    dx = np.zeros_like(gx)
    dy = np.zeros_like(gy)
    dx[:, 0]     =  gx[:, 0]
    dx[:, 1:-1]  =  gx[:, 1:-1] - gx[:, :-2]
    dx[:, -1]    = -gx[:, -2]
    dy[0, :]     =  gy[0, :]
    dy[1:-1, :]  =  gy[1:-1, :] - gy[:-2, :]
    dy[-1, :]    = -gy[-2, :]
    return dx + dy


def tv_prox_chambolle(b: np.ndarray, weight: float,
                      n_iter: int = 30) -> np.ndarray:
    """
    Solve  argmin_x  0.5 ||x - b||_2^2 + weight * TV(x)
    using Chambolle's projected-gradient algorithm on the dual.
    `b` must be 2-D.
    """
    # Chambolle (2004) dual projected-gradient algorithm.
    # Step size tau must satisfy tau <= 1/(4d) for d-dimensional TV.
    # In 2D the bound is tau <= 0.125; we use 0.1 with a safety margin.
    px = np.zeros_like(b)
    py = np.zeros_like(b)
    tau = 0.1

    for _ in range(n_iter):
        # gradient of the dual functional w.r.t. (px,py):  grad(div p - b/weight)
        div_p = _div_2d(px, py)
        gx, gy = _grad_2d(div_p - b / weight)
        px_new = px + tau * gx
        py_new = py + tau * gy
        # project onto the unit-ball constraint ||p||_inf <= 1
        denom = np.maximum(1.0, np.hypot(px_new, py_new))
        px = px_new / denom
        py = py_new / denom

    return b - weight * _div_2d(px, py)


def reconstruct_nesta_tv(Phi: np.ndarray, y: np.ndarray,
                         image_shape: tuple,
                         lam: float = 1e-2,
                         n_iter: int = 200,
                         tv_inner: int = 15,
                         L: float | None = None,
                         verbose: bool = False) -> np.ndarray:
    """
    Solve   argmin_x   0.5 ||Phi x - y||_2^2 + lam * TV(x)
    via FISTA (Nesterov-accelerated proximal gradient).  The TV proximal
    operator is computed with Chambolle's algorithm -- this gives a
    NESTA-style smoothing of the L1 norm on the image gradient.

    Parameters
    ----------
    Phi : (m, n) sensing matrix.
    y   : (m,) measurements.
    image_shape : (h, w), with h*w == n.
    lam : TV regularisation weight.
    n_iter : number of FISTA iterations.
    tv_inner : Chambolle inner iterations per TV prox.
    L   : Lipschitz constant of grad of 0.5||Phi x - y||^2 (= ||Phi||_2^2).
          If None, estimated via power iteration.
    """
    Phi = np.ascontiguousarray(Phi, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    h, w = image_shape
    n = h * w
    assert Phi.shape == (y.size, n)

    if L is None:
        # power iteration on Phi^T Phi
        v = np.random.RandomState(0).randn(n).astype(np.float32)
        v /= np.linalg.norm(v)
        for _ in range(30):
            v = Phi.T @ (Phi @ v)
            nv = np.linalg.norm(v)
            if nv == 0:
                break
            v /= nv
        L = float(np.linalg.norm(Phi.T @ (Phi @ v)) / max(np.linalg.norm(v), 1e-12))
        L *= 1.05                                 # safety margin

    step = 1.0 / L

    x      = np.zeros(n, dtype=np.float32)
    x_prev = x.copy()
    z      = x.copy()
    t_prev = 1.0

    for it in range(n_iter):
        grad = Phi.T @ (Phi @ z - y)
        u    = z - step * grad
        # TV proximal step on the reshaped image
        x    = tv_prox_chambolle(u.reshape(h, w), weight=lam * step,
                                 n_iter=tv_inner).ravel()
        t    = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_prev * t_prev))
        z    = x + ((t_prev - 1.0) / t) * (x - x_prev)
        x_prev, t_prev = x, t

        if verbose and (it % 25 == 0 or it == n_iter - 1):
            r = Phi @ x - y
            obj = 0.5 * float(r @ r) + lam * float(np.abs(np.diff(x.reshape(h, w), axis=0)).sum()
                                                   + np.abs(np.diff(x.reshape(h, w), axis=1)).sum())
            print(f"  iter {it:4d}   residual={np.linalg.norm(r):.4g}   obj={obj:.4g}")

    return x


# ---------------------------------------------------------------------------
# 5.  Quality metrics
# ---------------------------------------------------------------------------

def rmse(a: np.ndarray, b: np.ndarray, normalise: bool = True) -> float:
    """
    Root-mean-square error between two images.  If `normalise` is True,
    both inputs are scaled to [0, 1] before comparison (useful when one
    reconstruction is recovered only up to an unknown global scale).
    """
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if normalise:
        def n01(v):
            mn, mx = v.min(), v.max()
            return (v - mn) / (mx - mn + 1e-12)
        a = n01(a)
        b = n01(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def resize_image(img: np.ndarray, new_shape: tuple) -> np.ndarray:
    """Block-mean resize (image must be a multiple of new_shape)."""
    h, w = img.shape
    nh, nw = new_shape
    fh, fw = h // nh, w // nw
    return img[:nh*fh, :nw*fw].reshape(nh, fh, nw, fw).mean(axis=(1, 3))
