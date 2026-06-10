Nama : Aziza Hijran
NIM : F5512520040
Kelas : TI_A

Penyelesaian soal 11.1 sampai 11.28 dari buku Numerical Methods For Enginers menggunakan Python

============
Soal 11.1.py 
============
Metode: Algoritma Thomas (Eliminasi Gauss Khusus Matriks Tridiagonal).

Cara Kerja :  Berkas ini menyelesaikan persamaan matriks tridiagonal 3 x 3. Algoritma Thomas jauh lebih efisien dibanding eliminasi Gauss biasa karena ia hanya melakukan eliminasi (forward sweep) pada elemen diagonal bawah, lalu melakukan substitusi mundur (back substitution) untuk mencari akar persamaan.

Hasil: Solusi vektor $x$ berhasil ditemukan dengan cepat dari sistem persamaan diferensiasi tridiagonal tersebut.

============
Soal 11.2.py
============
Metode: Dekomposisi LU Manual (Tanpa Pivoting) & Invers Vektor Unit.
Cara Kerja: Matriks koefisien $A$ dipecah (factored) menjadi dua matriks segitiga: Matriks Segitiga Bawah (L / Lower) dan Matriks Segitiga Atas ($U$ / Upper) secara manual. Untuk menghitung nilai invers matriks (A^-1), program memasukkan beberapa vektor unit/identitas secara bergantian ke metode substitusi maju-mundur LU tersebut.
Hasil: Menghasilkan matriks L dan U yang valid serta keluaran matriks invers A^-1.

============
Soal_11.3.py
============
Metode: Algoritma Thomas untuk Sistem Skala Besar (Persamaan Crank-Nicolson).

Cara Kerja: Program ini menggunakan kembali struktur Algoritma Thomas modular. Namun, sistem yang diselesaikan di sini adalah matriks tridiagonal 5 x 5 yang biasanya merepresentasikan hasil diskretisasi persamaan diferensial parsial (PDE) stasioner menggunakan skema Crank-Nicolson.

Hasil: Menemukan profil solusi numerik (seperti temperatur atau konsentrasi T1 hingga T5) secara presisi dari skema Crank-Nicolson.

====================
Soal_11.4_ke_11.5.py
====================
Metode: Dekomposisi Cholesky Manual (A = L . L^T).

Cara Kerja: Metode ini digunakan khusus untuk matriks yang bersifat Simetri dan Definit Positif. Alih-alih memecahnya menjadi L dan U, Cholesky memecahnya menjadi matriks L dan transpose-nya sendiri (L^T). Soal 11.4 mengonfirmasi teori dari contoh buku, sementara Soal 11.5 menyelesaikan SPL simetri 3 X 3.

Hasil: Berhasil mengekstrak matriks akar kuadrat $L$, melakukan verifikasi perkalian L . L^T = A, dan mengeluarkan nilai solusi akhir vektor X.

=====================
Soal_11.6_ke_11.07.py
=====================
Metode: Dekomposisi Cholesky Lanjut & Kasus Matriks Diagonal.

Cara Kerja: Soal 11.6 menerapkan fungsi Cholesky manual pada matriks acak simetri lain untuk menguji tingkat residual error. Sedangkan Soal 11.7 membuktikan kasus khusus: jika matriksnya berupa matriks diagonal penuh, dekomposisi Cholesky menjadi sangat sederhana karena nilai elemen diagonal L cukup dicari dengan akar kuadrat langsung dari elemen diagonalA (akar A i,i).

Hasil: Menampilkan pembuktian teoretis Cholesky pada matriks diagonal dan solusi dengan nilai residual sangat kecil mendekati nol (2. 10^-14)

=====================
Soal_11.8_ke_11.11.py
=====================
Metode: Iterasi Gauss-Seidel dengan Overrelaksasi & Pengecekan Dominansi Diagonal.

Cara Kerja: Berbeda dengan metode langsung sebelumnya, ini adalah metode iteratif (perulangan). Program menebak nilai awal lalu memperbaruinya terus-menerus sampai error relatif persen berada di bawah toleransi (5%). Terdapat parameter relaksasi (1.2) untuk mempercepat lompatan konvergensi (over-relaxation).

Hasil: Menampilkan tabel baris per baris jalannya iterasi, status dominansi diagonal matriks, dan pembuktian bahwa sistem berhasil konvergen menuju solusi eksak.

======================
Soal_11.12_ke_11.15.py
======================
Metode: Analisis Konvergensi Gauss-Seidel & Pivot Penataan Ulang Baris Dominan.

Cara Kerja: Berkas ini berfokus pada masalah konvergensi Gauss-Seidel. Jika elemen diagonal utama tidak lebih besar dari jumlah elemen lainnya (tidak dominan diagonal), Gauss-Seidel akan gagal/divergen. Berkas ini memuat fungsi pintar susun_ulang_dominan() untuk menukar baris agar elemen terbesar menempati diagonal utama sebelum iterasi dimulai.

Hasil: Mengidentifikasi set matriks mana saja yang bisa dikerjakan (konvergen) dan mana yang tidak bisa diselesaikan oleh metode Gauss-Seidel.

=====================
Soal_11.16_ke11.17.py
=====================
Metode: Condition Number Matriks & Penyelesaian Sistem NONLINEAR (2D Newton/fsolve).

Cara Kerja: Soal 11.16 menghitung Condition Number menggunakan norma row-sum untuk mendeteksi apakah matriks bertipe ill-conditioned (sangat sensitif terhadap perubahan kecil). Soal 11.17 melompat ke sistem persamaan Nonlinear (f(x,y) = 4 - y - 2x^2) menggunakan modul scipy.optimize.fsolve.

Hasil: Memberikan kesimpulan kondisi matriks (Well vs Ill-conditioned). Untuk bagian nonlinear, program memplot grafik kontur perpotongan kurva menggunakan matplotlib dan menampilkan titik-titik solusi uniknya.

======================
Soal_11.18_ke_11.22.py
======================
Metode: Aplikasi SPL Dunia Nyata (Elektronik), Matriks Vandermonde, dan Matriks Augmented [A | I].

Cara Kerja: Berkas ini menerapkan pemecahan masalah produksi pabrik komponen (transistor, resistor, chip komputer) berdasarkan batas ketersediaan bahan baku (tembaga, seng, kaca) lewat np.linalg.solve. Selain itu, program mendemonstrasikan pembentukan matriks augmented gabungan [A | I] untuk mencari invers secara penuh.

Hasil: Menghasilkan angka rekomendasi jumlah unit produksi elektronik yang optimal, analisis matriks Vandermonde, dan hasil visualisasi invers gabungan.

======================
Soal_11.23_ke_11.28.py
======================
Metode: Pustaka Modular Campuran & Pemodelan Mass Balance Kanal 1D (Mekanis/Sipil/Kimia).

Cara Kerja: Berkas penutup ini mengemas seluruh algoritma (Thomas, Cholesky, Gauss-Seidel) menjadi fungsi modular yang siap pakai. Metode paling menarik di sini adalah mass_balance_canal(), yang menyelesaikan Persamaan Diferensial Parsial (Adveksi-Difusi-Peluruhan zat kimia di dalam kanal air) menggunakan pendekatan finite difference berpita (banded matrix).

Hasil: Berhasil menyelesaikan masalah rekayasa kanal, mencetak grafik profil penurunan konsentrasi zat kimia di sepanjang kanal