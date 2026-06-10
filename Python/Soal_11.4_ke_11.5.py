"""
Soal 11.4 & 11.5 - Dekomposisi Cholesky
=========================================

Rumus: A = L * L^T  (bukan L * U, tapi L * transpose-nya sendiri)

Soal 11.4: Konfirmasi dekomposisi Cholesky dari contoh 11.2
Soal 11.5: Sistem simetri:
[ 6   15   55 ] [u1]   [152.6 ]
[ 15  55  225 ] [u2] = [585.6 ]
[ 55 225  979 ] [u3]   [2488.8]
"""

import numpy as np


def cholesky_manual(A):
    """
    Dekomposisi Cholesky secara manual: A = L @ L.T
    
    Cara kerjanya:
    - isi matriks L baris per baris
    - Diagonal L dihitung dari akar kuadrat
    - Elemen bawah diagonal dihitung dari pembagian
    """
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    
    for i in range(n):
        for j in range(i + 1):
            total = sum(L[i, k] * L[j, k] for k in range(j))
            
            if i == j:
                nilai = A[i, i] - total
                if nilai <= 0:
                    raise ValueError(
                        f"Matriks tidak definit positif! "
                        f"Elemen diagonal [{i},{i}] = {nilai:.4f} ≤ 0"
                    )
                L[i, j] = np.sqrt(nilai)
            else:
                L[i, j] = (A[i, j] - total) / L[j, j]
    
    return L


def solve_cholesky(L, b):
    """
    Selesaikan Ax = b dengan A = L @ L.T
    
    Langkah:
      1. Ly  = b  → forward substitution
      2. L.T x = y → back substitution
    """
    n = len(b)
    
    # Forward: Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    
    # Backward: L.T @ x = y
    x = np.zeros(n)
    LT = L.T
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(LT[i, i + 1:], x[i + 1:])) / LT[i, i]
    
    return x


def main():
    print("=" * 55)
    print("Soal 11.4 - Konfirmasi Cholesky (Contoh 11.2)")
    print("=" * 55)
    
    # Matriks dari contoh 11.2 di buku
    A_11_4 = np.array([
        [2.0, -1.0,  0.0],
        [-1.0, 2.0, -1.0],
        [0.0, -1.0,  2.0]
    ])
    b_11_4 = np.array([1.0, 0.0, 1.0])
    
    print("\nMatriks A:")
    print(A_11_4)
    
    L = cholesky_manual(A_11_4)
    print("\nMatriks L (hasil Cholesky):")
    print(np.round(L, 6))
    
    print("\nVerifikasi L @ L.T (harus = A):")
    print(np.round(L @ L.T, 6))
    
    x = solve_cholesky(L, b_11_4)
    print(f"\nSolusi: {np.round(x, 6)}")
    print(f"Verifikasi A @ x: {np.round(A_11_4 @ x, 6)}")
    print(f"Vektor b asli:    {b_11_4}")
    
    # --------------------------------------------------
    print("\n" + "=" * 55)
    print("Soal 11.5 - Dekomposisi Cholesky Sistem Simetri")
    print("=" * 55)
    
    A_11_5 = np.array([
        [  6.,  15.,  55.],
        [ 15.,  55., 225.],
        [ 55., 225., 979.]
    ])
    b_11_5 = np.array([152.6, 585.6, 2488.8])
    
    print("\nMatriks A:")
    print(A_11_5)
    print("\nVektor b:", b_11_5)
    
    # untuk mengecek apakah matriks simetri
    print("\nApakah matriks simetri?", np.allclose(A_11_5, A_11_5.T))
    
    # Cek eigenvalue (semua harus positif untuk definit positif)
    eigenvalues = np.linalg.eigvalsh(A_11_5)
    print(f"Eigenvalue: {np.round(eigenvalues, 4)}")
    print(f"Semua positif? {np.all(eigenvalues > 0)}")
    
    L5 = cholesky_manual(A_11_5)
    print("\nMatriks L:")
    print(np.round(L5, 6))
    
    x5 = solve_cholesky(L5, b_11_5)
    print("\nSolusi:")
    for i, val in enumerate(x5):
        print(f"  u{i + 1} = {val:.6f}")
    
    # Verifikasi
    sisa = np.max(np.abs(A_11_5 @ x5 - b_11_5))
    print(f"\nResiduak maksimum: {sisa:.2e} ")


if __name__ == "__main__":
    main()
