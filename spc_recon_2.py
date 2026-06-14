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

def dmd_random_patterns(width, height, num_patterns=4000,
                        on_probability=0.5, macropixel=1):
    """
    Generate random binary DMD patterns at a given macropixel size.

    Each M x M block of micromirrors shares the same ON/OFF value, so the
    output is what the DMD physically displays for an M x M macropixel
    experiment.

    Parameters
    ----------
    width : int
        DMD width (number of micromirror columns).
    height : int
        DMD height (number of micromirror rows).
    num_patterns : int
        Number of random patterns to generate.
    on_probability : float
        Probability that a macropixel is ON (255).
    macropixel : int
        Macropixel size M.  width and height must be divisible by M.
        M = 1 reproduces the original per-mirror random pattern.

    Returns
    -------
    patterns : ndarray
        Shape: (num_patterns, height, width, 1)
        dtype: uint8
        Each M x M block is uniform (all 0 or all 255).
    """
    M = int(macropixel)
    if width % M != 0 or height % M != 0:
        raise ValueError(
            f"width ({width}) and height ({height}) must be divisible by "
            f"macropixel size M={M}"
        )

    # 1) Draw random ON/OFF values at the MACROPIXEL grid resolution.
    h_macro = height // M
    w_macro = width  // M
    small = np.random.choice(
        [0, 255],
        size=(num_patterns, h_macro, w_macro, 1),
        p=[1 - on_probability, on_probability]
    ).astype(np.uint8)

    # 2) Tile each macropixel value across its M x M block of micromirrors
    #    so the DMD physically displays uniform blocks.
    if M == 1:
        return small                                    # nothing to tile

    patterns = np.repeat(np.repeat(small, M, axis=1), M, axis=2)
    return patterns

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
#
# There are two distinct ways to obtain the sensing matrix at macropixel
# resolution M.  Use the one that matches how your data was acquired:
#
#   (a) build_phi_macropixel       <- PHYSICALLY CORRECT for an M x M experiment
#       Generates small binary patterns at macropixel resolution directly.
#       This is what the DMD really projects when each M x M block of
#       micromirrors is forced to the same ON/OFF state.
#
#   (b) build_sensing_matrix       <- re-interpretation of an existing 1 x 1 run
#       Takes patterns that were generated at full DMD resolution (every
#       micromirror independently random) and block-averages them down to the
#       macropixel grid.  This is NOT what a real M x M acquisition would
#       produce -- the rows are non-binary -- but it lets us reconstruct the
#       same single-pixel measurements at a coarser scale, under the
#       piecewise-constant scene assumption.
# ---------------------------------------------------------------------------

def build_phi_macropixel(N: int, h: int, w: int,
                         on_probability: float = 0.5,
                         seed: int = 42) -> np.ndarray:
    """
    Build the sensing matrix for a *true* macropixel experiment.

    For each pattern, every M x M block of micromirrors on the DMD takes the
    same binary value.  After grouping the measurement sum by blocks (see the
    derivation in the project notes), the per-block contribution to y_k is

        y_k  =  M^2 * sum_b  P_k^small(b) * x_b_mean ,

    so the sensing matrix at macropixel resolution is simply the stack of
    small binary patterns -- there is no need to tile them up to the DMD.

    Parameters
    ----------
    N : int
        Number of patterns (measurements).
    h, w : int
        Macropixel-grid shape  (h = H / M,  w = W / M).
    on_probability : float
        Probability that a macropixel is ON.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    Phi : ndarray of shape (N, h * w), float32, entries in {0, 1}.
    """
    rng = np.random.default_rng(seed)
    Phi = (rng.random((N, h * w), dtype=np.float32) < on_probability)
    return Phi.astype(np.float32)


def build_sensing_matrix(patterns: np.ndarray,
                         macropixel: int,
                         normalise: bool = True) -> np.ndarray:
    """
    Form a sensing matrix from previously acquired 1 x 1 patterns, by
    block-averaging them down to macropixel resolution.

    Use this when you have a *single* dataset of measurements acquired with
    1 x 1 random patterns and want to reconstruct it at several coarser
    scales without re-running the experiment.  The resulting Phi rows are
    NOT binary; entries lie in [0, 1].

    For a clean from-scratch simulation of an M x M macropixel experiment,
    use ``build_phi_macropixel`` instead.

    Parameters
    ----------
    patterns : (N, H, W) array
        Binary patterns at full DMD resolution (values 0/255 or 0/1).
    macropixel : int
        Macropixel size M.
    normalise : bool
        If True, divide by 255 so entries are in [0, 1] regardless of input dtype.

    Returns
    -------
    Phi : (N, (H/M)*(W/M)) float32
    """
    P = crop_to_multiple(patterns, macropixel)
    P_macro = block_average(P, macropixel)         # (N, h, w)  float32
    N = P_macro.shape[0]
    Phi = P_macro.reshape(N, -1)
    if normalise:
        Phi = Phi / 255.0
    return Phi


def make_pattern_operator(patterns: np.ndarray,
                          batch: int = 100,
                          normalise: bool = True) -> LinearOperator:
    """
    Build a *matrix-free* sensing operator from a stack of binary patterns.

    At full DMD resolution (1140 x 912, N = 2000) the explicit float32
    sensing matrix would be ~8 GB.  This operator keeps the patterns as
    uint8 (~2 GB) and converts batches to float32 on the fly when computing
    ``Phi @ x`` or ``Phi.T @ v``.

    The returned object behaves exactly like a numpy array for the purposes
    of ``Phi @ x``, ``Phi.T @ v``, ``Phi.shape``, and ``Phi.dtype`` -- so it
    can be passed straight to ``scipy.sparse.linalg.lsqr`` or to
    ``reconstruct_nesta_tv``.

    Parameters
    ----------
    patterns : (N, H, W) uint8 array, binary values 0 or 255.
    batch    : how many patterns to convert to float32 at a time.
               Larger = faster but more peak memory.
    normalise: if True, the operator treats entries as 0/1 instead of 0/255.

    Returns
    -------
    Phi_op : scipy.sparse.linalg.LinearOperator of shape (N, H*W).
    """
    N, H, W = patterns.shape
    n_pix = H * W
    scale = np.float32(1.0 / 255.0 if normalise else 1.0)

    def matvec(x):
        x2d = np.asarray(x, dtype=np.float32).reshape(H, W)
        out = np.empty(N, dtype=np.float32)
        for k in range(0, N, batch):
            chunk = patterns[k:k + batch].astype(np.float32) * scale  # (b,H,W)
            out[k:k + batch] = (chunk * x2d).sum(axis=(1, 2))
        return out

    def rmatvec(v):
        v = np.asarray(v, dtype=np.float32)
        out = np.zeros((H, W), dtype=np.float32)
        for k in range(0, N, batch):
            b = min(batch, N - k)
            chunk = patterns[k:k + b].astype(np.float32) * scale       # (b,H,W)
            out += (chunk * v[k:k + b, None, None]).sum(axis=0)
        return out.ravel()

    return LinearOperator((N, n_pix), matvec=matvec, rmatvec=rmatvec,
                          dtype=np.float32)


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
    Phi : (m, n) sensing matrix OR scipy.sparse.linalg.LinearOperator.
    y   : (m,) measurements.
    image_shape : (h, w), with h*w == n.
    lam : TV regularisation weight.
    n_iter : number of FISTA iterations.
    tv_inner : Chambolle inner iterations per TV prox.
    L   : Lipschitz constant of grad of 0.5||Phi x - y||^2 (= ||Phi||_2^2).
          If None, estimated via power iteration.
    """
    # Accept either dense matrix or LinearOperator
    if not isinstance(Phi, LinearOperator):
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
