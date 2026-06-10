"""
Soal 11.12 - 11.15: Konvergensi dan Relaksasi Gauss-Seidel
===========================================================
Soal 11.12: Gauss-Seidel TANPA dan DENGAN relaksasi (λ = 0.95)
           untuk sistem:
           -3x1 + x2 + 12x3 = 50
            6x1 - x2 - x3   = 3
            6x1 + 9x2 + x3  = 40

Soal 11.13: Gauss-Seidel dengan relaksasi (λ = 1.2) untuk:
            2x1 - 6x2 - x3  = -38
           -3x1 + x2 + 7x3  = -34
           -8x1 + x2 - 2x3  = -20

Soal 11.14: Kondisi di mana Gauss-Seidel TIDAK konvergen
            (ketika lereng garis sama, slope = 1)

Soal 11.15: Identifikasi sistem yang tidak bisa diselesaikan Gauss-Seidel
"""

import numpy as np


def gauss_seidel(A, b, x0=None, tol=0.05, maks_iter=200, lam=1.0, verbose=True):
    """Gauss-Seidel dengan relaksasi opsional."""
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    riwayat = []
    
    for it in range(1, maks_iter + 1):
        x_lama = x.copy()
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_baru = (b[i] - sigma) / A[i, i]
            x[i] = lam * x_baru + (1 - lam) * x_lama[i]
        
        # Error relatif persen
        denom = np.where(np.abs(x) > 1e-10, np.abs(x), 1e-10)
        err = np.max(np.abs(x - x_lama) / denom * 100)
        riwayat.append(err)
        
        if verbose and it <= 10:
            vals = "  ".join(f"x{i+1}={x[i]:8.4f}" for i in range(n))
            print(f"  Iter {it:2d}: {vals}  | err={err:.4f}%")
        elif verbose and it == 11:
            print("  ...")
        
        if err < tol:
            if verbose:
                print(f"\n✓ Konvergen pada iterasi ke-{it}")
            return x, riwayat, it
    
    if verbose:
        print(f"\n Tidak konvergen setelah {maks_iter} iterasi")
    return x, riwayat, maks_iter


def susun_ulang_dominan(A, b):
    """
   Menyusun ulang bais agar diagonal dominan.
 dengan menukar baris sehingga elemen terbesar ada di diagonal.
    """
    n = len(b)
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    
    for i in range(n):
        maks_baris = i
        for k in range(i + 1, n):
            if abs(A[k, i]) > abs(A[maks_baris, i]):
                maks_baris = k
        A[[i, maks_baris]] = A[[maks_baris, i]]
        b[[i, maks_baris]] = b[[maks_baris, i]]
    
    return A, b


def main():
    # ===== SOAL 11.12 =====
    print("=" * 65)
    print("Soal 11.12 - Gauss-Seidel dengan/tanpa Relaksasi")
    print("=" * 65)
    
    A12_asli = np.array([
        [-3.,  1., 12.],
        [ 6., -1., -1.],
        [ 6.,  9.,  1.]
    ])
    b12 = np.array([50., 3., 40.])
    
    # Cek apakah perlu susun ulang
    print("(a) Menyusun ulang agar dominan diagonal:")
    A12, b12r = susun_ulang_dominan(A12_asli, b12)
    print(f"  Matriks setelah disusun ulang:\n{A12}")
    print(f"  b = {b12r}")
    
    print("\n  Gauss-Seidel TANPA relaksasi (λ = 1.0):")
    x12a, _, it12a = gauss_seidel(A12, b12r, tol=5.0, lam=1.0)
    
    print("\n(b) Dengan relaksasi λ = 0.95:")
    x12b, _, it12b = gauss_seidel(A12, b12r, tol=5.0, lam=0.95)
    
    x_eks = np.linalg.solve(A12, b12r)
    print(f"\nSolusi eksak: {np.round(x_eks, 4)}")
    print(f"Iterasi tanpa relaksasi: {it12a}  |  Dengan λ=0.95: {it12b}")
    
    # ===== SOAL 11.13 =====
    print("\n" + "=" * 65)
    print("Soal 11.13 - Gauss-Seidel dengan Relaksasi λ = 1.2")
    print("=" * 65)
    
    A13_asli = np.array([
        [ 2., -6., -1.],
        [-3.,  1.,  7.],
        [-8.,  1., -2.]
    ])
    b13 = np.array([-38., -34., -20.])
    
    A13, b13r = susun_ulang_dominan(A13_asli, b13)
    print(f"Setelah susun ulang:\n{A13}")
    
    print("\n  Tanpa relaksasi:")
    x13a, _, it13a = gauss_seidel(A13, b13r, tol=5.0, lam=1.0)
    
    print("\n  Dengan overrelaksasi λ = 1.2:")
    x13b, _, it13b = gauss_seidel(A13, b13r, tol=5.0, lam=1.2)
    
    x13_eks = np.linalg.solve(A13, b13r)
    print(f"\nSolusi eksak: {np.round(x13_eks, 4)}")
    
    # ===== SOAL 11.14 =====
    print("\n" + "=" * 65)
    print("Soal 11.14 - Kapan Gauss-Seidel TIDAK Konvergen?")
    print("=" * 65)
    print("""
Gauss-Seidel tidak akan konvergen kalau matriks tidak
dominan diagonal. Contoh sistem yang TIDAK konvergen:

  x1 + 2*x2 = 3   ← diagonal = 1, off-diagonal = 2 → TIDAK dominan
  3*x1 + x2 = 4

Kalau |A[i,i]| < jumlah |A[i,j]| untuk j ≠ i,
metode iteratif biasanya diverge (solusi malah makin jauh).
    """)
    
    A14_tidak = np.array([[1., 2.], [3., 1.]])
    b14 = np.array([3., 4.])
    
    print("Sistem tidak konvergen (diagonal tidak dominan):")
    print("  Gauss-Seidel dengan maks 20 iterasi:")
    x14, riwayat14, it14 = gauss_seidel(A14_tidak, b14, tol=0.001, maks_iter=20)
    
    if max(riwayat14) > 100:
        print("  → Error membesar = DIVERGEN (tidak konvergen) ✓")
    
    print("\nSolusi eksak (lewat numpy):", np.round(np.linalg.solve(A14_tidak, b14), 4))
    
    # ===== SOAL 11.15 =====
    print("\n" + "=" * 65)
    print("Soal 11.15 - Identifikasi Sistem yang Tidak Bisa Gauss-Seidel")
    print("=" * 65)
    
    sets = {
        "Set 1": {
            "A": np.array([[8., 3., -7.], [-2., 1., 5.], [4., -3., -2.]]),
            "b": np.array([7., 4., 5.])
        },
        "Set 2": {
            "A": np.array([[1., 4., -1.], [-3., 1., 2.], [1., 1., 6.]]),
            "b": np.array([7., -3., 7.])
        },
        "Set 3": {
            "A": np.array([[-1., 3., 5.], [3., -7., 2.], [1., 2., -5.]]),
            "b": np.array([7., 2., -1.])
        }
    }
    
    for nama, data in sets.items():
        A_s, b_s = data["A"], data["b"]
        n = len(b_s)
        print(f"\n  {nama}:")
        dominan = True
        for i in range(n):
            diag = abs(A_s[i, i])
            off = sum(abs(A_s[i, j]) for j in range(n) if j != i)
            if diag <= off:
                dominan = False
        
        if not dominan:
            A_sorted, b_sorted = susun_ulang_dominan(A_s, b_s)
            dominan2 = True
            for i in range(n):
                diag = abs(A_sorted[i, i])
                off = sum(abs(A_sorted[i, j]) for j in range(n) if j != i)
                if diag <= off:
                    dominan2 = False
            status = "Setelah susun ulang: dominan ✓" if dominan2 else "Tetap tidak dominan → sulit konvergen"
        else:
            status = "Dominan diagonal ✓"
        print(f"    {status}")


if __name__ == "__main__":
    main()
