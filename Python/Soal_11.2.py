"""
Soal 11.2 - Dekomposisi LU dan Vektor Unit
==========================================
Tentukan invers matriks menggunakan dekomposisi LU
dari contoh 11.1. Dengan pecah matriks A menjadi
dua matriks segitiga L (bawah) dan U (atas), lalu
gunakan untuk menghitung invers.

Matriks yang digunakan sama seperti soal 11.1:
[ 0.8  -0.4   0  ]
[-0.4   0.8  -0.4]
[ 0    -0.4   0.8]
"""

import numpy as np
from scipy.linalg import lu


def lu_decomposition_manual(A):
    """
    Dekomposisi LU secara manual tanpa pivoting.
    
    Ide dasarnya: kita tulis A = L * U
    di mana L adalah matriks segitiga bawah (lower)
    dan U adalah matriks segitiga atas (upper).
    """

    n = A.shape[0]
    L = np.eye(n)       # mulai dengan matriks identitas
    U = A.copy().astype(float)
    
    for k in range(n - 1):
        for i in range(k + 1, n):
            if U[k, k] == 0:
                raise ValueError("Pivot nol! Perlu pivoting.")
            faktor = U[i, k] / U[k, k]
            L[i, k] = faktor
            U[i, k:] -= faktor * U[k, k:]
    
    return L, U


def solve_dengan_lu(L, U, b):
    """
    Selesaikan Ax=b setelah mendapat L dan U.
    Langkahnya:
      1. Ly = b  → forward substitution 
      2. Ux = y  → back substitution 
    """
    n = len(b)
    
    # Forward substitution: Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    # Back substitution: Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    
    return x


def hitung_invers_dengan_lu(A):
    """
    Hitung invers matriks menggunakan LU.
    """
    n = A.shape[0]
    L, U = lu_decomposition_manual(A)
    A_inv = np.zeros((n, n))
    
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0   # vektor unit
        A_inv[:, i] = solve_dengan_lu(L, U, e)
    
    return L, U, A_inv


def main():
    print("=" * 55)
    print("Soal 11.2 - Dekomposisi LU & Invers Matriks")
    print("=" * 55)
    
    A = np.array([
        [ 0.8, -0.4,  0.0],
        [-0.4,  0.8, -0.4],
        [ 0.0, -0.4,  0.8]
    ])
    
    print("\nMatriks A:")
    print(A)
    
    L, U, A_inv = hitung_invers_dengan_lu(A)
    
    print("\nMatriks L (segitiga bawah):")
    print(np.round(L, 6))
    
    print("\nMatriks U (segitiga atas):")
    print(np.round(U, 6))
    
    print("\nInvers A menggunakan LU:")
    print(np.round(A_inv, 6))
    
    # Verifikasi: A @ A_inv harus menghasilkan matriks identitas
    produk = A @ A_inv
    print("\nVerifikasi A @ A_inv (harus = I):")
    print(np.round(produk, 6))
    
    # Bandingkan dengan numpy
    A_inv_np = np.linalg.inv(A)
    print("\nInvers dari numpy (pembanding):")
    print(np.round(A_inv_np, 6))
    
    selisih = np.max(np.abs(A_inv - A_inv_np))
    print(f"\nSelisih maksimum: {selisih:.2e} ")


if __name__ == "__main__":
    main()
