"""
utils/helpers.py
================
Fungsi-fungsi pembantu yang dipakai bersama di berbagai soal.
Kamu bisa import dari sini supaya kode tidak duplikat.

Contoh:
    from utils.helpers import gauss_seidel, thomas_algorithm, cholesky
"""

import numpy as np


def thomas_algorithm(sub, main_d, sup, rhs):
    """
    Algoritma Thomas - solusi cepat untuk sistem tridiagonal.
    Kompleksitas O(n) jauh lebih efisien dari O(n³) eliminasi Gauss biasa.
    """
    n = len(rhs)
    sub    = sub.copy().astype(float)
    main_d = main_d.copy().astype(float)
    sup    = sup.copy().astype(float)
    rhs    = rhs.copy().astype(float)
    
    for i in range(1, n):
        m = sub[i] / main_d[i - 1]
        main_d[i] -= m * sup[i - 1]
        rhs[i]    -= m * rhs[i - 1]
    
    x = np.zeros(n)
    x[-1] = rhs[-1] / main_d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (rhs[i] - sup[i] * x[i + 1]) / main_d[i]
    return x


def cholesky(A):
    """Dekomposisi Cholesky: A = L @ L.T (hanya untuk matriks simetri definit positif)."""
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                v = A[i, i] - s
                if v <= 0:
                    raise ValueError(f"Tidak definit positif di [{i},{i}]: {v:.6f}")
                L[i, j] = np.sqrt(v)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L


def cholesky_solve(L, b):
    """Selesaikan Ax=b setelah A = L @ L.T"""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(L.T[i, i + 1:], x[i + 1:])) / L.T[i, i]
    return x


def gauss_seidel(A, b, tol_pct=5.0, max_iter=200, lam=1.0, x0=None):
    """
    Gauss-Seidel iteratif dengan relaksasi.
    
    Returns: (solusi, daftar_error, jumlah_iterasi)
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    errors = []
    
    for it in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new = (b[i] - sigma) / A[i, i]
            x[i] = lam * x_new + (1 - lam) * x_old[i]
        
        denom = np.where(np.abs(x) > 1e-12, np.abs(x), 1e-12)
        err = np.max(np.abs(x - x_old) / denom * 100)
        errors.append(err)
        
        if err < tol_pct:
            return x, errors, it
    
    return x, errors, max_iter


def is_diagonally_dominant(A):
    """Cek apakah matriks dominan diagonal (syarat konvergensi Gauss-Seidel)."""
    n = A.shape[0]
    for i in range(n):
        if abs(A[i, i]) <= sum(abs(A[i, j]) for j in range(n) if j != i):
            return False
    return True


def condition_number_inf(A):
    """Hitung condition number menggunakan norma infinity (row-sum norm)."""
    A_inv = np.linalg.inv(A)
    norm_A   = np.max(np.sum(np.abs(A), axis=1))
    norm_inv = np.max(np.sum(np.abs(A_inv), axis=1))
    return norm_A * norm_inv
