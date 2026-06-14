"""
spc_cs_l1.py
============
Compressed-sensing reconstruction for the single-pixel camera using an
*explicit sparsifying basis* Psi (DCT or Haar wavelets) + L1 minimisation.

This is the recovery written on the lecture slides:

        y = Phi f ,      f = Psi alpha       (f is K-sparse in Psi)

    reconstruct by solving the basis-pursuit-denoising (LASSO) problem

        min_alpha  1/2 || Phi Psi alpha - y ||_2^2  +  lam * || alpha ||_1
        then        f_hat = Psi alpha_hat

Compared with the TV solver you already have, here the prior is "few large
DCT / wavelet coefficients" instead of "small gradient". Different Psi ->
different reconstruction from the *same* measurements (slide 25).

Solver: FISTA (accelerated proximal gradient) with soft-thresholding.
Everything is matrix-free in Psi (uses fast DCT / DWT), so it scales to the
macropixel grids you reconstruct on.

Author: built to plug into Project6_Notebook_*.ipynb
"""

import numpy as np
from scipy.fft import dctn, idctn

try:
    import pywt
    _HAVE_PYWT = True
except Exception:                                    # pragma: no cover
    _HAVE_PYWT = False


# --------------------------------------------------------------------------
# Sparsifying bases Psi.  Each returns a pair of callables:
#   syn(alpha_vec) -> f_vec        (synthesis:  f = Psi alpha)
#   ana(f_vec)     -> alpha_vec    (analysis:   alpha = Psi^T f)
# For orthonormal Psi (both options below) ana == syn^{-1} == syn^T, which is
# exactly what FISTA needs for the gradient step.
# `dc_mask` marks coefficients that should NOT be L1-penalised (the DC term
# for DCT, the coarse approximation band for Haar) -- important here because
# most of your y is a DC offset and you don't want to shrink the mean away.
# --------------------------------------------------------------------------
def _dct_basis(shape):
    h, w = shape

    def syn(a):
        return idctn(a.reshape(h, w), norm="ortho").ravel()

    def ana(f):
        return dctn(f.reshape(h, w), norm="ortho").ravel()

    dc_mask = np.zeros(h * w, dtype=bool)
    dc_mask[0] = True                                # (0,0) coefficient = DC
    return syn, ana, dc_mask


def _haar_basis(shape, level=None):
    if not _HAVE_PYWT:
        raise ImportError("Haar basis needs PyWavelets: pip install pywavelets")
    h, w = shape
    wavelet = "haar"
    mode = "periodization"                           # keeps the transform orthonormal
    if level is None:
        level = pywt.dwtn_max_level((h, w), wavelet)

    # Build the coefficient<->vector bookkeeping once, on a dummy image.
    coeffs0 = pywt.wavedec2(np.zeros((h, w)), wavelet, mode=mode, level=level)
    arr0, slices = pywt.coeffs_to_array(coeffs0)
    arr_shape = arr0.shape

    def syn(a):
        arr = a.reshape(arr_shape)
        coeffs = pywt.array_to_coeffs(arr, slices, output_format="wavedec2")
        return pywt.waverec2(coeffs, wavelet, mode=mode).ravel()

    def ana(f):
        coeffs = pywt.wavedec2(f.reshape(h, w), wavelet, mode=mode, level=level)
        arr, _ = pywt.coeffs_to_array(coeffs)
        return arr.ravel()

    # The coarse approximation band sits in the top-left corner of the array.
    aslice = slices[0]                               # tuple of slices for cA
    mask2d = np.zeros(arr_shape, dtype=bool)
    mask2d[aslice] = True
    dc_mask = mask2d.ravel()
    return syn, ana, dc_mask


def _get_basis(basis, shape, level=None):
    basis = basis.lower()
    if basis in ("dct", "cosine"):
        return _dct_basis(shape)
    if basis in ("haar", "wavelet", "db1"):
        return _haar_basis(shape, level=level)
    raise ValueError(f"unknown basis {basis!r}; use 'dct' or 'haar'")


# --------------------------------------------------------------------------
# Wrap Phi (dense ndarray OR scipy LinearOperator) into matvec / rmatvec.
# --------------------------------------------------------------------------
def _phi_ops(Phi):
    if hasattr(Phi, "matvec") and hasattr(Phi, "rmatvec"):
        return (lambda x: Phi.matvec(x)), (lambda v: Phi.rmatvec(v)), Phi.shape
    Phi = np.asarray(Phi)
    return (lambda x: Phi @ x), (lambda v: Phi.T @ v), Phi.shape


def _power_iteration(matA, matAt, n, iters=60, seed=0):
    """Largest eigenvalue of A^T A (= ||A||_2^2), for the FISTA step size."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x /= np.linalg.norm(x)
    lam = 1.0
    for _ in range(iters):
        x = matAt(matA(x))
        lam = np.linalg.norm(x)
        if lam == 0:
            break
        x /= lam
    return lam


def _soft(x, thr):
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)


# --------------------------------------------------------------------------
# Main entry point.
# --------------------------------------------------------------------------
def reconstruct_l1(Phi, y, image_shape, basis="dct",
                   lam=None, lam_frac=0.02, n_iter=400,
                   penalize_dc=False, level=None,
                   nonneg=False, verbose=False):
    """
    Solve   min_alpha 1/2||Phi Psi alpha - y||^2 + lam ||alpha||_1   via FISTA,
    then return f_hat = Psi alpha reshaped to `image_shape`.

    Parameters
    ----------
    Phi : (M, N) ndarray or scipy LinearOperator
        Sensing matrix at the reconstruction grid (N = prod(image_shape)).
        Use the same Phi you feed to your Tikhonov / TV solvers.
    y : (M,) array
        Bucket measurements.
    image_shape : (h, w)
        Reconstruction grid (e.g. the macropixel grid h_macro x w_macro).
    basis : {'dct', 'haar'}
        Sparsifying transform Psi.
    lam : float or None
        L1 weight. If None, set lam = lam_frac * ||Psi^T Phi^T y||_inf, the
        scale above which the all-zero solution is optimal -- a robust,
        problem-scaled default. Lower lam_frac -> sharper / noisier.
    lam_frac : float
        Fraction used when lam is None (default 0.02 ~ 2%).
    n_iter : int
        FISTA iterations.
    penalize_dc : bool
        If False (default) the DC / coarse-approximation coefficients are not
        L1-shrunk -- recommended here because y is DC-heavy.
    nonneg : bool
        Project f >= 0 each step (scene intensities are non-negative).
    verbose : bool
        Print objective every ~10% of iterations.

    Returns
    -------
    f_hat : (h, w) ndarray
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    h, w = image_shape
    N = h * w
    syn, ana, dc_mask = _get_basis(basis, image_shape, level=level)
    matPhi, matPhiT, (M, Ncol) = _phi_ops(Phi)
    if Ncol != N:
        raise ValueError(f"Phi has {Ncol} columns but image_shape implies {N}")

    # Composite operator A = Phi Psi  and its transpose A^T = Psi^T Phi^T.
    coeff_dim = ana(np.zeros(N)).size                # = N for dct, >= N for haar padding
    A = lambda a: matPhi(syn(a))
    At = lambda r: ana(matPhiT(r))

    # Step size from Lipschitz constant L = ||A||_2^2.
    L = _power_iteration(A, At, coeff_dim)
    L = max(L, 1e-12)
    step = 1.0 / L

    # which coefficients to threshold (DC / coarse band left unpenalised)
    pen = np.ones(coeff_dim) if penalize_dc else (~dc_mask).astype(float)

    # lambda default, scaled to the data. IMPORTANT: take the max over the
    # *penalised* coefficients only -- otherwise the (huge) DC coefficient of
    # A^T y inflates lambda and thresholds every AC coefficient to zero.
    if lam is None:
        g = np.abs(At(y))
        g_pen = g[pen > 0] if np.any(pen > 0) else g
        lam = lam_frac * np.max(g_pen)
    thr = lam * step

    alpha = np.zeros(coeff_dim)
    z = alpha.copy()
    t = 1.0
    for k in range(n_iter):
        r = A(z) - y                                 # residual
        grad = At(r)
        alpha_new = z - step * grad
        alpha_new = _soft(alpha_new, thr * pen)      # prox of L1 (DC untouched)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = alpha_new + ((t - 1.0) / t_new) * (alpha_new - alpha)
        alpha, t = alpha_new, t_new

        if nonneg:
            f = np.maximum(syn(alpha), 0.0)          # re-project to feasible image
            alpha = ana(f)
            z = alpha.copy()

        if verbose and (k % max(1, n_iter // 10) == 0 or k == n_iter - 1):
            data = 0.5 * np.sum((A(alpha) - y) ** 2)
            l1 = np.sum(np.abs(alpha * pen))
            print(f"  iter {k:4d}  data={data:.4g}  l1={l1:.4g}  "
                  f"obj={data + lam * l1:.4g}")

    f_hat = syn(alpha).reshape(h, w)
    if nonneg:
        f_hat = np.maximum(f_hat, 0.0)
    return f_hat


# Convenience wrappers --------------------------------------------------------
def reconstruct_dct_l1(Phi, y, image_shape, **kw):
    return reconstruct_l1(Phi, y, image_shape, basis="dct", **kw)


def reconstruct_haar_l1(Phi, y, image_shape, **kw):
    return reconstruct_l1(Phi, y, image_shape, basis="haar", **kw)
