"""
Soal 11.8 - 11.11: Gauss-Seidel dan Jacobi
============================================

Soal 11.8:  Gauss-Seidel untuk sistem dari soal 11.1 (tridiagonal)
Soal 11.9:  Gauss-Seidel dengan overrelaksasi λ = 1.2
Soal 11.10: Hitung jumlah operasi Gauss eliminasi tanpa pivoting
Soal 11.11: Gauss-Seidel untuk sistem 3x3
"""

import numpy as np
import matplotlib.pyplot as plt


def gauss_seidel(A, b, x0=None, toleransi=0.05, maks_iter=100, lam=1.0, verbose=True):
    """
    Metode Gauss-Seidel untuk menyelesaikan Ax = b.
    
    Parameter:
        A        : matriks koefisien
        b        : vektor ruas kanan
        x0       : tebakan awal (default: vektor nol)
        toleransi: error relatif maksimum yang diijinkan (%)
        maks_iter: batas maksimum iterasi
        lam      : faktor relaksasi (1.0 = tanpa relaksasi,
                   > 1.0 = over-relaxation, < 1.0 = under-relaxation)
        verbose  : tampilkan setiap iterasi atau tidak
    
    Return:
        x         : solusi
        riwayat   : daftar error setiap iterasi
        n_iterasi : jumlah iterasi yang dilakukan
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    riwayat = []
    
    if verbose:
        header = "Iter | " + " ".join(f"  x{i+1:>8}" for i in range(n)) + " | Error maks (%)"
        print(header)
        print("-" * len(header))
    
    for iterasi in range(1, maks_iter + 1):
        x_lama = x.copy()
        
        for i in range(n):
            # Hitung nilai baru untuk x[i]
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_baru_i = (b[i] - sigma) / A[i, i]
            
            # Terapkan faktor relaksasi
            x[i] = lam * x_baru_i + (1 - lam) * x_lama[i]
        
        # Hitung error relatif (hindari bagi nol)
        error = np.max(
            np.abs((x - x_lama) / np.where(x != 0, x, 1e-10)) * 100
        )
        riwayat.append(error)
        
        if verbose:
            vals = " ".join(f"{v:10.5f}" for v in x)
            print(f"  {iterasi:2d}  | {vals} | {error:8.4f}")
        
        if error < toleransi:
            if verbose:
                print(f"\n✓ Konvergen setelah {iterasi} iterasi (error = {error:.4f}% < {toleransi}%)")
            return x, riwayat, iterasi
    
    if verbose:
        print(f"\n⚠ Belum konvergen setelah {maks_iter} iterasi")
    return x, riwayat, maks_iter


def main():
    # ===== SOAL 11.8 =====
    print("=" * 65)
    print("Soal 11.8 - Gauss-Seidel pada Sistem Tridiagonal")
    print("=" * 65)
    
    A8 = np.array([
        [ 0.8, -0.4,  0.0],
        [-0.4,  0.8, -0.4],
        [ 0.0, -0.4,  0.8]
    ])
    b8 = np.array([41.0, 25.0, 105.0])
    
    # Untuk Gauss-Seidel, kita perlu cek dominansi diagonal
    # Matriks ini dominan diagonal? (|A[i,i]| > jumlah |A[i,j]| untuk j≠i)
    for i in range(3):
        diag = abs(A8[i, i])
        off_diag = sum(abs(A8[i, j]) for j in range(3) if j != i)
        print(f"  Baris {i+1}: |{diag}| {'>' if diag > off_diag else '<='} {off_diag} → {'dominan ✓' if diag > off_diag else 'tidak dominan'}")
    
    print("\nHasil iterasi Gauss-Seidel:")
    x8, riwayat8, _ = gauss_seidel(A8, b8, toleransi=5.0)
    
    # ===== SOAL 11.9 =====
    print("\n" + "=" * 65)
    print("Soal 11.9 - Gauss-Seidel dengan Overrelaksasi (λ = 1.2)")
    print("=" * 65)
    print("""
Overrelaksasi (λ > 1): kita 'mendorong lebih jauh' dari nilai baru.
Tujuannya: mempercepat konvergensi.
Ibaratnya: kalau kamu bergerak ke arah benar, ambil langkah lebih besar.
    """)
    
    print("Tanpa relaksasi (λ = 1.0):")
    x_no_rel, riwayat_no_rel, iter_no_rel = gauss_seidel(
        A8, b8, toleransi=5.0, lam=1.0, verbose=False
    )
    print(f"  Jumlah iterasi: {iter_no_rel}")
    
    print("\nDengan overrelaksasi (λ = 1.2):")
    x_rel, riwayat_rel, iter_rel = gauss_seidel(
        A8, b8, toleransi=5.0, lam=1.2, verbose=False
    )
    print(f"  Jumlah iterasi: {iter_rel}")
    
    if iter_rel < iter_no_rel:
        print(f"\n→ Overrelaksasi lebih cepat ({iter_no_rel} → {iter_rel} iterasi) ✓")
    else:
        print(f"\n→ Overrelaksasi tidak selalu lebih cepat untuk kasus ini")
    
    # ===== SOAL 11.10 =====
    print("\n" + "=" * 65)
    print("Soal 11.10 - Jumlah Operasi Eliminasi Gauss")
    print("=" * 65)
    print("""
Untuk matriks n×n, eliminasi Gauss (tanpa pivoting) butuh:
  - Forward elimination: ~ (2n³/3) operasi
  - Back substitution : ~ n² operasi
  - Total             : ~ 2n³/3 + n² ≈ 2n³/3 (untuk n besar)
    """)
    
    for n in [3, 5, 10, 20, 100]:
        ops_elim = int(2 * n**3 / 3)
        ops_subs = n**2
        total = ops_elim + ops_subs
        print(f"  n = {n:3d}  |  Forward: {ops_elim:8d}  |  Substitusi: {ops_subs:6d}  |  Total: {total:8d}")
    
    # ===== SOAL 11.11 =====
    print("\n" + "=" * 65)
    print("Soal 11.11 - Gauss-Seidel Sistem 3x3 (toleransi 5%)")
    print("=" * 65)
    
    A11 = np.array([
        [10.,  2.,  1.],
        [ 2., 10., -1.],
        [ 1.,  1.,  5.]
    ])
    b11 = np.array([27., 0., 21.5])
    
    # Cek dominansi diagonal
    print("Cek dominansi diagonal:")
    dominan = True
    for i in range(3):
        diag = abs(A11[i, i])
        off  = sum(abs(A11[i, j]) for j in range(3) if j != i)
        status = diag > off
        dominan = dominan and status
        print(f"  Baris {i+1}: {diag} {'>' if status else '<='} {off} → {'✓' if status else '✗'}")
    
    print(f"\nDominan diagonal: {'Ya → akan konvergen' if dominan else 'Tidak'}")
    
    print("\nHasil iterasi:")
    x11, riwayat11, iter11 = gauss_seidel(A11, b11, toleransi=5.0)
    
    # Solusi eksak dari numpy
    x_eksak = np.linalg.solve(A11, b11)
    print("\nSolusi eksak (numpy):")
    for i, v in enumerate(x_eksak):
        print(f"  x{i+1} = {v:.6f}")
    
    # Plot konvergensi
    plt.figure(figsize=(8, 4))
    plt.semilogy(riwayat11, 'b-o', markersize=4, label='Error (%)')
    plt.axhline(y=5.0, color='r', linestyle='--', label='Toleransi 5%')
    plt.xlabel("Iterasi")
    plt.ylabel("Error Relatif (%)")
    plt.title("Konvergensi Gauss-Seidel (Soal 11.11)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("konvergensi_gauss_seidel_11_11.png", dpi=100)
    plt.close()
    print("\nGrafik konvergensi disimpan: konvergensi_gauss_seidel_11_11.png")


if __name__ == "__main__":
    main()
