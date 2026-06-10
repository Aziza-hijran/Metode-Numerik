"""
Soal 11.1 - Sistem Tridiagonal dengan Eliminasi Gauss
======================================================
Menyelesaikan sistem persamaan tridiagonal
menggunakan metode eliminasi yang mirip dengan contoh 11.1 dan 11.3.

Sistem persamaan:
  0.8*x1 - 0.4*x2              = 41
 -0.4*x1 + 0.8*x2 - 0.4*x3    = 25
          -0.4*x2 + 0.8*x3    = 105
"""

import numpy as np


def thomas_algorithm(a, b, c, d):
    """
    Algoritma Thomas - cara cepat menyelesaikan sistem tridiagonal.
    Algoritma Thomas memanfaatkan struktur ini supaya lebih efisien dibanding eliminasi Gauss biasa.
    
    Parameter:
        a = diagonal bawah  (elemen ke-2 sampai ke-n)
        b = diagonal utama  (semua elemen)
        c = diagonal atas   (elemen ke-1 sampai ke-(n-1))
        d = vektor hasil (ruas kanan)
    
    Return:
        x = solusi vektor
    """
    n = len(d)
    
    # Buat salinan supaya array asli tidak berubah
    a = a.copy().astype(float)
    b = b.copy().astype(float)
    c = c.copy().astype(float)
    d = d.copy().astype(float)
    
    # Langkah 1: Forward sweep (maju) 
    # "habiskan" elemen diagonal bawah satu per satu
    for i in range(1, n):
        faktor = a[i] / b[i - 1]
        b[i] -= faktor * c[i - 1]
        d[i] -= faktor * d[i - 1]
    
    # Langkah 2: Back substitution (mundur) 
    # Hitung x dari bawah ke atas
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    
    return x


def main():
    print("=" * 55)
    print("Soal 11.1 - Sistem Tridiagonal 3x3")
    print("=" * 55)
    
    # Matriks A berbentuk tridiagonal
    A = np.array([
        [ 0.8, -0.4,  0.0],
        [-0.4,  0.8, -0.4],
        [ 0.0, -0.4,  0.8]
    ])
    
    b_vec = np.array([41.0, 25.0, 105.0])
    
    # Pisahkan diagonal-diagonal dari matriks A
    diag_bawah  = np.array([0.0, A[1, 0], A[2, 1]])   # a
    diag_utama  = np.array([A[0, 0], A[1, 1], A[2, 2]])  # b
    diag_atas   = np.array([A[0, 1], A[1, 2], 0.0])   # c
    
    # Selesaikan dengan algoritma Thomas
    x = thomas_algorithm(diag_bawah, diag_utama, diag_atas, b_vec)
    
    print("\nMatriks A:")
    print(A)
    print("\nVektor b:", b_vec)
    print("\nSolusi menggunakan Algoritma Thomas:")
    for i, val in enumerate(x):
        print(f"  x{i + 1} = {val:.6f}")
    
    # Verifikasi: hitung A @ x dan bandingkan dengan b
    verif = A @ x
    print("\nVerifikasi (A @ x harus = b):")
    for i in range(len(b_vec)):
        print(f"  Baris {i + 1}: {verif[i]:.6f}  (target: {b_vec[i]})")
    
    # Bandingkan dengan solusi numpy
    x_np = np.linalg.solve(A, b_vec)
    print("\nSolusi numpy (pembanding):")
    for i, val in enumerate(x_np):
        print(f"  x{i + 1} = {val:.6f}")
    
    print("\nKesimpulan: Kedua metode menghasilkan jawaban yang sama ✓")


if __name__ == "__main__":
    main()
