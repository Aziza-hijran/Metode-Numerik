"""
Soal 11.23 - 11.28: Aplikasi Lanjut - Thomas, Cholesky, PDE, Vandermonde
=========================================================================
Soal 11.23: Program Thomas algorithm untuk sistem tridiagonal
Soal 11.24: Program Cholesky (duplikasi hasil contoh 11.2)
Soal 11.25: Program Gauss-Seidel (duplikasi hasil contoh 11.3)
Soal 11.26: Masa balance untuk kanal 1D (persamaan diferensial parsial)
Soal 11.27: Persamaan diferensial dengan PDE steady-state (mass balance)
Soal 11.28: Sistem pentadiagonal (bandwidth 5)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded


# ============================================================
# Soal 11.23 - Algoritma Thomas sebagai fungsi modular
# ============================================================
def thomas_algorithm(sub, main_d, sup, rhs):
    """
    Algoritma Thomas yang siap dipakai ulang (reusable).
    
    sub    = diagonal bawah (elemen [1..n-1])
    main_d = diagonal utama (elemen [0..n-1])
    sup    = diagonal atas  (elemen [0..n-2])
    rhs    = vektor kanan
    """
    n = len(rhs)
    sub    = sub.copy().astype(float)
    main_d = main_d.copy().astype(float)
    sup    = sup.copy().astype(float)
    rhs    = rhs.copy().astype(float)
    
    # Forward sweep
    for i in range(1, n):
        m = sub[i] / main_d[i - 1]
        main_d[i] -= m * sup[i - 1]
        rhs[i]    -= m * rhs[i - 1]
    
    # Back substitution
    x = np.zeros(n)
    x[-1] = rhs[-1] / main_d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (rhs[i] - sup[i] * x[i + 1]) / main_d[i]
    
    return x


# ============================================================
# Soal 11.24 - Fungsi Cholesky modular
# ============================================================
def cholesky_decomp(A):
    """Dekomposisi Cholesky: A = L @ L.T"""
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                v = A[i, i] - s
                if v <= 0:
                    raise ValueError("Matriks tidak definit positif")
                L[i, j] = np.sqrt(v)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L


def cholesky_solve(L, b):
    """Solusi sistem Ax=b dengan A = L @ L.T"""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(L.T[i, i + 1:], x[i + 1:])) / L.T[i, i]
    return x


# ============================================================
# Soal 11.25 - Fungsi Gauss-Seidel modular
# ============================================================
def gauss_seidel_solver(A, b, tol=0.05, max_iter=100, lam=1.0):
    """Gauss-Seidel dengan relaksasi opsional."""
    n = len(b)
    x = np.zeros(n)
    for it in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new = (b[i] - sigma) / A[i, i]
            x[i] = lam * x_new + (1 - lam) * x_old[i]
        err = np.max(np.abs(x - x_old) / np.where(np.abs(x) > 1e-10, np.abs(x), 1e-10) * 100)
        if err < tol:
            return x, it + 1
    return x, max_iter


# ============================================================
# Soal 11.26 - Mass Balance Kanal 1D (PDE stasioner)
# ============================================================
def mass_balance_canal(n_nodes=20, delta_x=2.0, D=2.0, U=1.0, k=0.2,
                        c0=0.0, cn=10.0):
    """
    Selesaikan persamaan mass balance stasioner di kanal 1D:
    
      0 = D * d²c/dx² - U * dc/dx - k * c
    
    di mana:
      c = konsentrasi bahan kimia
      D = koefisien difusi
      U = kecepatan aliran
      k = laju peluruhan (first-order)
    
    Diskretisasi dengan finite difference menghasilkan sistem tridiagonal.
    
    Kondisi batas:
      c(0)   = c0   (batas kiri)
      c(L)   = cn   (batas kanan)
    """
    dx = delta_x
    
    # Koefisien finite difference
    a_i = D / dx**2 + U / (2 * dx)   # koef baris i-1
    b_i = -2 * D / dx**2 - k         # koef diagonal
    c_i = D / dx**2 - U / (2 * dx)   # koef baris i+1
    
    # Bangun sistem tridiagonal
    n = n_nodes
    sub     = np.full(n, a_i)
    main_d  = np.full(n, b_i)
    sup     = np.full(n, c_i)
    rhs     = np.zeros(n)
    
    # Terapkan kondisi batas
    rhs[0]    -= a_i * c0
    rhs[-1]   -= c_i * cn
    sub[0]    = 0.0
    sup[-1]   = 0.0
    
    c_sol = thomas_algorithm(sub, main_d, sup, rhs)
    
    # Tambahkan node batas
    x_full = np.arange(0, (n + 2) * dx, dx)
    c_full = np.concatenate([[c0], c_sol, [cn]])
    
    return x_full, c_full


def main():
    # ===== SOAL 11.23 =====
    print("=" * 65)
    print("Soal 11.23 - Algoritma Thomas (Program Modular)")
    print("=" * 65)
    
    # Uji dengan sistem dari contoh 11.2 buku
    sub    = np.array([0., -0.4, -0.4])
    main_d = np.array([0.8, 0.8, 0.8])
    sup    = np.array([-0.4, -0.4, 0.])
    rhs    = np.array([41., 25., 105.])
    
    x23 = thomas_algorithm(sub, main_d, sup, rhs)
    print("\nSistem tridiagonal soal 11.1 diselesaikan ulang:")
    for i, v in enumerate(x23):
        print(f"  x{i+1} = {v:.6f}")
    
    # Uji dengan sistem lebih besar
    print("\nUji Thomas untuk n=5:")
    n5 = 5
    sub5    = np.full(n5, -1.)
    main5   = np.full(n5,  2.)
    sup5    = np.full(n5, -1.)
    rhs5    = np.array([1., 0., 0., 0., 1.])
    x5      = thomas_algorithm(sub5, main5, sup5, rhs5)
    print(f"  Solusi: {np.round(x5, 6)}")
    
    # ===== SOAL 11.24 =====
    print("\n" + "=" * 65)
    print("Soal 11.24 - Cholesky (Duplikasi Contoh 11.2)")
    print("=" * 65)
    
    A24 = np.array([
        [ 6.,  15.,  55.],
        [15.,  55., 225.],
        [55., 225., 979.]
    ])
    b24 = np.array([152.6, 585.6, 2488.8])
    
    L24 = cholesky_decomp(A24)
    x24 = cholesky_solve(L24, b24)
    
    print("Matriks L:")
    print(np.round(L24, 6))
    print(f"\nSolusi: {np.round(x24, 6)}")
    print(f"Verifikasi: {np.round(A24 @ x24, 4)} ≈ {b24}")
    
    # ===== SOAL 11.25 =====
    print("\n" + "=" * 65)
    print("Soal 11.25 - Gauss-Seidel (Duplikasi Contoh 11.3)")
    print("=" * 65)
    
    A25 = np.array([
        [10., 2., -1.],
        [-3., -6., 2.],
        [1.,  1.,  5.]
    ])
    b25 = np.array([27., -61.5, 21.5])
    
    x25, iters = gauss_seidel_solver(A25, b25, tol=1.0, max_iter=50)
    print(f"Konvergen dalam {iters} iterasi")
    print(f"Solusi: {np.round(x25, 4)}")
    print(f"Eksak : {np.round(np.linalg.solve(A25, b25), 4)}")
    
    # ===== SOAL 11.26 - 11.27 =====
    print("\n" + "=" * 65)
    print("Soal 11.26 & 11.27 - Mass Balance Kanal 1D")
    print("=" * 65)
    print("""
PDE stasioner:  0 = D*d²c/dx² - U*dc/dx - k*c
  D = 2.0 (difusi)
  U = 1.0 (kecepatan)
  k = 0.2 (peluruhan)
  c(0) = 0,  c(L) = 10
    """)
    
    x26, c26 = mass_balance_canal(n_nodes=20, delta_x=2.0, D=2.0, U=1.0, k=0.2)
    
    print("Profil konsentrasi:")
    for xi, ci in zip(x26[::4], c26[::4]):
        bar = '█' * int(ci / 10 * 20)
        print(f"  x = {xi:5.1f} | c = {ci:6.4f} | {bar}")
    
    plt.figure(figsize=(9, 4))
    plt.plot(x26, c26, 'b-o', markersize=5, linewidth=2)
    plt.xlabel("Posisi x (m)")
    plt.ylabel("Konsentrasi c")
    plt.title("Soal 11.26 - Profil Konsentrasi Kanal 1D\n(D=2, U=1, k=0.2)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mass_balance_kanal_11_26.png", dpi=100)
    plt.close()
    print("\nPlot profil konsentrasi: mass_balance_kanal_11_26.png")
    
    # ===== SOAL 11.28 =====
    print("\n" + "=" * 65)
    print("Soal 11.28 - Sistem Pentadiagonal")
    print("=" * 65)
    print("""
Sistem pentadiagonal = bandwidth 5 (2 di atas + diagonal + 2 di bawah).
Muncul dalam metode numerik orde tinggi atau elemen hingga.

Kita gunakan sistem dari soal sebagai contoh.
    """)
    
    A28 = np.array([
        [ 8., -2.,  1.,  0.,  0.,  0.],
        [-2.,  9., -4.,  1.,  0.,  0.],
        [ 1., -4.,  7., -1.,  2.,  0.],
        [ 0.,  1., -1., 12., -5.,  1.],
        [ 0.,  0.,  2., -5.,  9., -2.],
        [ 0.,  0.,  0.,  1., -2.,  5.]
    ])
    b28 = np.array([5., 2., 0., 1., 0., 1.])
    
    print("Matriks Pentadiagonal (6x6):")
    print(A28)
    
    x28 = np.linalg.solve(A28, b28)
    print(f"\nSolusi: {np.round(x28, 6)}")
    print(f"Verifikasi residual: {np.max(np.abs(A28 @ x28 - b28)):.2e} ✓")
    
    print("\n" + "=" * 65)
    print("RINGKASAN SEMUA SOAL SELESAI ✓")
    print("=" * 65)


if __name__ == "__main__":
    main()
