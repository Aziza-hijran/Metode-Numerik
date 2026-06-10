"""
Soal 11.3 - Sistem Tridiagonal Besar (Crank-Nicolson)
======================================================

Kita gunakan kembali Algoritma Thomas yang sudah terbukti efisien.
Contoh sistem:
[ 2.01475  -0.020875   0         0         0       ] [T1]   [4.175 ]
[-0.020875  2.01475  -0.020875   0         0       ] [T2] = [0     ]
[ 0        -0.020875  2.01475  -0.020875   0       ] [T3]   [0     ]
[ 0         0        -0.020875  2.01475  -0.020875 ] [T4]   [0     ]
[ 0         0         0        -0.020875  2.01475  ] [T5]   [2.0875]
"""

import numpy as np


def thomas_algorithm(a, b, c, d):
    """Algoritma Thomas untuk sistem tridiagonal."""
    n = len(d)
    a = a.copy().astype(float)
    b = b.copy().astype(float)
    c = c.copy().astype(float)
    d = d.copy().astype(float)
    
    # Forward sweep
    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]
    
    # Back substitution
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    
    return x


def main():
    print("=" * 60)
    print("Soal 11.3 - Sistem Tridiagonal (Crank-Nicolson)")
    print("=" * 60)
    
    # Nilai dari soal 
    diagonal_utama = np.array([2.01475, 2.01475, 2.01475, 2.01475, 2.01475])
    diagonal_atas  = np.array([-0.020875, -0.020875, -0.020875, -0.020875, 0.0])
    diagonal_bawah = np.array([0.0, -0.020875, -0.020875, -0.020875, -0.020875])
    rhs            = np.array([4.175, 0.0, 0.0, 0.0, 2.0875])
    
    print("\nInput sistem:")
    print(f"  Diagonal utama : {diagonal_utama}")
    print(f"  Diagonal atas  : {diagonal_atas}")
    print(f"  Diagonal bawah : {diagonal_bawah}")
    print(f"  Ruas kanan     : {rhs}")
    
    x = thomas_algorithm(diagonal_bawah, diagonal_utama, diagonal_atas, rhs)
    
    print("\nSolusi (T1 hingga T5):")
    for i, val in enumerate(x):
        print(f"  T{i + 1} = {val:.6f}")
    
    # Membangun matriks penuh untuk verifikasi
    n = len(rhs)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = diagonal_utama[i]
        if i > 0:
            A[i, i - 1] = diagonal_bawah[i]
        if i < n - 1:
            A[i, i + 1] = diagonal_atas[i]
    
    residual = np.max(np.abs(A @ x - rhs))
    print(f"\nResiduak maksimum: {residual:.2e}")
    print("Semakin kecil residual, semakin tepat solusinya ")


if __name__ == "__main__":
    main()
