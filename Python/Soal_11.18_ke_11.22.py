"""
Soal 11.18 - 11.22: Aplikasi Matriks & Perintah Python/NumPy
=============================================================
Soal 11.18: Masalah produksi elektronik (transistor, resistor, chip)
            → selesaikan dengan sistem linear A @ x = b

Soal 11.19: Spektral dekomposisi (kondisi num. matriks Hilbert 10D)

Soal 11.20: Validasi hasil dengan condition number untuk Vandermonde

Soal 11.21: Buat [Aaug] = [A | I] (augmented matrix)

Soal 11.22: Sistem persamaan linear sederhana
"""

import numpy as np
from scipy.linalg import hilbert


def main():
    # ===== SOAL 11.18 =====
    print("=" * 65)
    print("Soal 11.18 - Produksi Elektronik (Transistor, Resistor, Chip)")
    print("=" * 65)
    print("""
Tabel komponen per unit produk:
              Tembaga  Seng  Kaca
Transistor      4       3     2
Resistor        3       3     1
Chip komputer   2       1     3

Ketersediaan bahan baku minggu ini (dalam satuan):
  Tembaga : 960
  Seng    : 510
  Kaca    : 610

Kita cari: berapa unit transistor, resistor, dan chip yang dibuat?
    """)
    
    # Susun sebagai sistem Ax = b
    # x = [transistor, resistor, chip]
    A18 = np.array([
        [4., 3., 2.],   # Tembaga: 4T + 3R + 2C = 960
        [3., 3., 1.],   # Seng:    3T + 3R + 1C = 510
        [2., 1., 3.]    # Kaca:    2T + 1R + 3C = 610
    ])
    b18 = np.array([960., 510., 610.])
    
    x18 = np.linalg.solve(A18, b18)
    
    print("Matriks koefisien A:")
    print(A18)
    print(f"\nBahan baku b: {b18}")
    print("\nSolusi produksi:")
    nama_produk = ["Transistor", "Resistor  ", "Chip      "]
    for nama, val in zip(nama_produk, x18):
        print(f"  {nama}: {val:.1f} unit")
    
    print(f"\nVerifikasi (A @ x = b):")
    print(f"  {np.round(A18 @ x18, 2)} ≈ {b18} ✓")
    
    # ===== SOAL 11.19 =====
    print("\n" + "=" * 65)
    print("Soal 11.19 - Kondisi Matriks Hilbert 10 Dimensi")
    print("=" * 65)
    print("""
Matriks Hilbert adalah contoh klasik matriks ILL-CONDITIONED.
Elemen H[i,j] = 1 / (i + j - 1)

Semakin besar ukurannya, semakin 'berbahaya' (condition number makin besar).
    """)
    
    for n in [3, 5, 8, 10]:
        H = hilbert(n)
        cond = np.linalg.cond(H)
        print(f"  Hilbert {n}x{n}: condition number ≈ {cond:.3e}")
    
    H10 = hilbert(10)
    print(f"\nElemen pertama matriks Hilbert 10x10:")
    print(np.round(H10[:4, :4], 6))
    
    # Berapa digit presisi yang hilang?
    print(f"\nEstimasi digit presisi yang hilang (log10 cond):")
    print(f"  ≈ {np.log10(np.linalg.cond(H10)):.1f} digit")
    print("  Ini sangat besar! Solusinya tidak bisa dipercaya.")
    
    # ===== SOAL 11.20 =====
    print("\n" + "=" * 65)
    print("Soal 11.20 - Matriks Vandermonde 5D")
    print("=" * 65)
    print("""
Matriks Vandermonde: V[i,j] = x_i^(j-1)
Digunakan untuk interpolasi polinomial.
    """)
    
    x_pts = np.array([1., 2., 3., 4., 5.])
    n = len(x_pts)
    V = np.vander(x_pts, increasing=True)  # V[i,j] = x[i]^j
    
    print(f"Titik: {x_pts}")
    print("\nMatriks Vandermonde:")
    print(V)
    
    cond_v = np.linalg.cond(V)
    print(f"\nCondition number: {cond_v:.3e}")
    
    if cond_v > 1e6:
        print("⚠ Matriks Vandermonde cenderung ill-conditioned untuk n besar")
    
    # ===== SOAL 11.21 =====
    print("\n" + "=" * 65)
    print("Soal 11.21 - Augmented Matrix [A | I]")
    print("=" * 65)
    print("""
Di MATLAB: [Aaug] = [A, eye(n)]
Di Python/NumPy: np.hstack([A, np.eye(n)])

Augmented matrix dipakai untuk menghitung invers dengan eliminasi Gauss.
    """)
    
    A21 = np.array([
        [2., 1., -1.],
        [-3., -1., 2.],
        [-2.,  1., 2.]
    ])
    
    I = np.eye(3)
    Aaug = np.hstack([A21, I])
    
    print("Matriks A:")
    print(A21)
    print("\nAugmented [A | I]:")
    print(np.round(Aaug, 2))
    
    # Eliminasi Gauss pada Aaug
    Aaug_copy = Aaug.copy()
    for k in range(3):
        for i in range(k + 1, 3):
            m = Aaug_copy[i, k] / Aaug_copy[k, k]
            Aaug_copy[i, :] -= m * Aaug_copy[k, :]
    
    # Back substitution
    for k in range(2, -1, -1):
        Aaug_copy[k, :] /= Aaug_copy[k, k]
        for i in range(k):
            Aaug_copy[i, :] -= Aaug_copy[i, k] * Aaug_copy[k, :]
    
    A_inv_calc = Aaug_copy[:, 3:]
    print("\nInvers A (dari eliminasi Gauss pada augmented):")
    print(np.round(A_inv_calc, 6))
    print("\nInvers dari numpy:")
    print(np.round(np.linalg.inv(A21), 6))
    
    # ===== SOAL 11.22 =====
    print("\n" + "=" * 65)
    print("Soal 11.22 - Sistem Persamaan Linear dari Matriks")
    print("=" * 65)
    
    A22 = np.array([
        [ 1., 7., 0., 0.],
        [ 4., 7., 3., 0.],
        [ 1., -7., 3., 5.],
        [ 0., 4., -7., 5.]
    ])
    b22 = np.array([50., 0., 40., 0.])
    
    print("Sistem 50 = x1 + 7x2 :")
    print("Matriks A:")
    print(A22)
    
    x22 = np.linalg.solve(A22, b22)
    print(f"\nSolusi: {np.round(x22, 4)}")
    print(f"Verifikasi: {np.round(A22 @ x22, 4)} ≈ {b22}")


if __name__ == "__main__":
    main()
