"""
Soal 11.16 - 11.17: Invers Matriks, Condition Number, Sistem Nonlinear
=======================================================================
Soal 11.16: Hitung invers dan condition number dua matriks:
            (a) Matriks 3x3 Hilbert-like
            (b) Matriks 4x4

Soal 11.17: Sistem persamaan NONLINEAR (bukan linear lagi!):
            f(x,y) = 4 - y - 2x²
            g(x,y) = 8 - y² - 4x
            → Gunakan Excel Solver atau metode Newton 2D
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def condition_number_row_sum(A):
    """
    Hitung condition number menggunakan norma row-sum (norma tak-hingga).
     
    - Condition number ≈ 1  → sistem baik (well-conditioned)
    - Condition number >> 1 → sistem buruk (ill-conditioned)
    """
    A_inv = np.linalg.inv(A)
    
    # Norma row-sum = nilai maksimum dari jumlah |elemen| per baris
    norma_A   = np.max(np.sum(np.abs(A), axis=1))
    norma_inv = np.max(np.sum(np.abs(A_inv), axis=1))
    
    cond = norma_A * norma_inv
    return A_inv, cond


def main():
    # ===== SOAL 11.16(a) =====
    print("=" * 60)
    print("Soal 11.16(a) - Invers & Condition Number Matriks 3x3")
    print("=" * 60)
    
    A_a = np.array([
        [1., 4., 9.],
        [4., 9., 16.],
        [9., 16., 25.]
    ])
    b_a = np.array([14., 29., 50.])
    
    print("\nMatriks A:")
    print(A_a)
    
    A_inv_a, cond_a = condition_number_row_sum(A_a)
    
    print("\nInvers A:")
    print(np.round(A_inv_a, 6))
    
    print(f"\nCondition number (norma row-sum): {cond_a:.4f}")
    
    if cond_a > 1000:
        print(" Condition number besar → sistem ILL-CONDITIONED")
    
    else:
        print("✓ Condition number kecil → sistem well-conditioned")
    
    x_a = np.linalg.solve(A_a, b_a)
    print(f"\nSolusi x: {np.round(x_a, 6)}")
    
    # ===== SOAL 11.16(b) =====
    print("\n" + "=" * 60)
    print("Soal 11.16(b) - Invers & Condition Number Matriks 4x4")
    print("=" * 60)
    
    A_b = np.array([
        [ 1.,  4.,  9., 16.],
        [ 4.,  9., 16., 25.],
        [ 9., 16., 25., 36.],
        [16., 25., 36., 49.]
    ])
    b_b = np.array([30., 54., 86., 126.])
    
    print("\nMatriks A:")
    print(A_b)
    
    A_inv_b, cond_b = condition_number_row_sum(A_b)
    
    print("\nInvers A:")
    print(np.round(A_inv_b, 4))
    
    print(f"\nCondition number: {cond_b:.4f}")
    print(f"Condition number numpy (pembanding): {np.linalg.cond(A_b, np.inf):.4f}")
    
    x_b = np.linalg.solve(A_b, b_b)
    print(f"\nSolusi x: {np.round(x_b, 6)}")
    print("(Semua x harusnya = 1)")
    
    # ===== SOAL 11.17 =====
    print("\n" + "=" * 60)
    print("Soal 11.17 - Sistem Persamaan NONLINEAR 2D")
    print("=" * 60)
    print("""
Sistem:
  f(x, y) = 4 - y - 2x²  = 0
  g(x, y) = 8 - y² - 4x = 0

Untuk mencari solusi, kita coba Newton 2D (mirip Newton-Raphson
tapi untuk lebih dari 1 variabel). Jacobian dibutuhkan:

  J = [ df/dx  df/dy ] = [ -4x   -1  ]
      [ dg/dx  dg/dy ]   [ -4   -2y  ]
    """)
    
    def sistem(vars_):
        x, y = vars_
        f = 4 - y - 2 * x**2
        g = 8 - y**2 - 4 * x
        return [f, g]
    
    # Cari semua solusi dengan berbagai tebakan awal
    tebakan_awal_list = [
        (-3., -3.), (-3., 3.), (3., -3.), (3., 3.),
        (0., 2.), (1., 1.), (-1., 3.), (2., -2.)
    ]
    
    solusi_unik = []
    print("Mencari solusi dengan berbagai tebakan awal:")
    
    for tebakan in tebakan_awal_list:
        try:
            sol = fsolve(sistem, tebakan, full_output=True)
            x_sol, info, ier, msg = sol
            if ier == 1:  # konvergen
                # Cek apakah solusi ini sudah ditemukan sebelumnya
                baru = True
                for s in solusi_unik:
                    if np.allclose(x_sol, s, atol=1e-4):
                        baru = False
                        break
                if baru:
                    solusi_unik.append(x_sol)
                    residual = np.max(np.abs(sistem(x_sol)))
                    print(f"  Tebakan ({tebakan[0]:5.1f},{tebakan[1]:5.1f}) → "
                          f"Solusi: x={x_sol[0]:.6f}, y={x_sol[1]:.6f}  "
                          f"(residual={residual:.2e})")
        except Exception:
            pass
    
    print(f"\nTotal solusi unik ditemukan: {len(solusi_unik)}")
    
    # Visualisasi kurva
    x_plot = np.linspace(-4, 4, 300)
    y_plot = np.linspace(-4, 4, 300)
    X, Y = np.meshgrid(x_plot, y_plot)
    F = 4 - Y - 2 * X**2
    G = 8 - Y**2 - 4 * X
    
    plt.figure(figsize=(7, 6))
    plt.contour(X, Y, F, levels=[0], colors='blue', linewidths=2, label='f=0')
    plt.contour(X, Y, G, levels=[0], colors='red', linewidths=2, label='g=0')
    
    for i, s in enumerate(solusi_unik):
        plt.plot(s[0], s[1], 'ko', markersize=10, zorder=5)
        plt.annotate(f"S{i+1}\n({s[0]:.2f},{s[1]:.2f})",
                     xy=(s[0], s[1]), xytext=(s[0]+0.3, s[1]+0.3),
                     fontsize=9)
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Soal 11.17 - Sistem Nonlinear\nBiru: f=0, Merah: g=0")
    plt.grid(True, alpha=0.3)
    plt.legend(["f(x,y)=0", "g(x,y)=0"])
    plt.tight_layout()
    plt.savefig("solusi_nonlinear_11_17.png", dpi=100)
    plt.close()
    print("\nPlot kurva disimpan: solusi_nonlinear_11_17.png")


if __name__ == "__main__":
    main()
