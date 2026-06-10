"""
Soal 11.6 & 11.7 - Dekomposisi Cholesky Lanjut
================================================
Soal 11.6: Lakukan Cholesky secara manual untuk:
[8   20  15]       [50 ]
[20  80  50] x = [250]
[15  50  60]       [100]

Soal 11.7: Hitung Cholesky dari matriks diagonal:
[9  0  0]
[0  25 0]
[0  0  4]
(Matriks diagonal simetri definit positif – kasus paling sederhana!)
"""

import numpy as np


def cholesky_manual(A):
    """Dekomposisi Cholesky: A = L @ L.T"""
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                val = A[i, i] - s
                if val <= 0:
                    raise ValueError(f"Tidak definit positif pada [{i},{i}]")
                L[i, j] = np.sqrt(val)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L


def solve_cholesky(L, b):
    """Selesaikan sistem Ax=b setelah A = L @ L.T"""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    x = np.zeros(n)
    LT = L.T
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(LT[i, i + 1:], x[i + 1:])) / LT[i, i]
    return x


def main():
    # ===== SOAL 11.6 =====
    print("=" * 55)
    print("Soal 11.6 - Cholesky Manual pada Sistem 3x3")
    print("=" * 55)
    
    A6 = np.array([
        [ 8., 20., 15.],
        [20., 80., 50.],
        [15., 50., 60.]
    ])
    b6 = np.array([50., 250., 100.])
    
    print("\nMatriks A:")
    print(A6)
    
    # untuk mengecek apakah valid untuk Cholesky
    eigenval = np.linalg.eigvalsh(A6)
    print(f"\nEigenvalue: {np.round(eigenval, 4)}")
    print(f"Semua positif (cocok untuk Cholesky)? {np.all(eigenval > 0)}")
    
    L6 = cholesky_manual(A6)
    print("\nMatriks L:")
    print(np.round(L6, 6))
    
    print("\nVerifikasi L @ L.T:")
    print(np.round(L6 @ L6.T, 4))
    
    x6 = solve_cholesky(L6, b6)
    print("\nSolusi:")
    for i, v in enumerate(x6):
        print(f"  x{i+1} = {v:.6f}")
    
    print(f"\nResiduak: {np.max(np.abs(A6 @ x6 - b6)):.2e} ")
    
    # ===== SOAL 11.7 =====
    print("\n" + "=" * 55)
    print("Soal 11.7 - Cholesky Matriks Diagonal")
    print("=" * 55)
    print("""
Untuk matriks diagonal, Cholesky sangat mudah:
  L[i,i] = sqrt(A[i,i])
  L[i,j] = 0  untuk i ≠ j

Karena semua elemen off-diagonal = 0.
    """)
    
    A7 = np.array([
        [9., 0., 0.],
        [0., 25., 0.],
        [0., 0., 4.]
    ])
    
    print("Matriks A (diagonal):")
    print(A7)
    
    L7 = cholesky_manual(A7)
    print("\nMatriks L:")
    print(np.round(L7, 6))
    
    print("\nPenjelasan intuitif:")
    print(f"  sqrt(9)  = {np.sqrt(9):.4f}")
    print(f"  sqrt(25) = {np.sqrt(25):.4f}")
    print(f"  sqrt(4)  = {np.sqrt(4):.4f}")
    
    print("\nVerifikasi L @ L.T = A:")
    print(np.round(L7 @ L7.T, 4))
    print("\nCholesky matriks diagonal = akar kuadrat dari tiap elemen diagonal")


if __name__ == "__main__":
    main()
