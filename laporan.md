# Laporan Proyek Machine Learning - Sistem Rekomendasi Film Berbasis Hybrid Filtering

**Nama:** Robby
**Program:** Machine Learning Engineering
**Tanggal:** Juni 2025

---

## Domain Proyek

Dalam era digital saat ini, berbagai platform streaming seperti Netflix, Disney+, dan Amazon Prime menyediakan ribuan judul film untuk penggunanya. Fenomena ini mengarah pada masalah klasik *information overload*, di mana pengguna kesulitan dalam memilih film yang sesuai dengan preferensi mereka. Oleh karena itu, sistem rekomendasi menjadi komponen penting untuk meningkatkan kenyamanan pengguna dan mendorong keterlibatan lebih lanjut.

Menurut McKinsey Global Institute, sistem rekomendasi menyumbang sekitar 35% dari total revenue Amazon dan 75% content yang ditonton di Netflix berasal dari rekomendasi. Sistem ini tidak hanya meningkatkan user retention tetapi juga memberikan keuntungan kompetitif bagi perusahaan.

**Mengapa masalah ini penting:**

1. Meningkatkan user experience dengan memberikan saran personal.
2. Mengurangi waktu pencarian film yang sesuai.
3. Meningkatkan retensi dan nilai bisnis platform.
4. Mengatasi *cold-start problem* baik pada user maupun item.

---

## Business Understanding

### Problem Statements:

1. Bagaimana menyarankan film yang mirip dengan yang pernah disukai oleh pengguna berdasarkan genre dan statistik rating?
2. Bagaimana memberikan saran film baru yang belum pernah ditonton dengan memanfaatkan data pengguna lain yang memiliki pola serupa?
3. Bagaimana menggabungkan dua pendekatan rekomendasi untuk meningkatkan relevansi dan personalisasi?

### Goals:

1. Mengembangkan sistem rekomendasi berbasis konten menggunakan informasi film.
2. Mengembangkan sistem rekomendasi berbasis kolaboratif (Collaborative Filtering).
3. Menghasilkan top-N rekomendasi untuk setiap user berdasarkan pendekatan tersebut.

### Solution Statements:

* **Content-Based Filtering:** Menggunakan informasi fitur film seperti genre, rating rata-rata, dan varian rating untuk mengukur kesamaan antar item.
* **Collaborative Filtering:** Menggunakan matrix factorization (SVD) untuk memprediksi nilai rating antar user dan film yang belum dirating.

---

## Data Understanding

### Informasi Dataset:

* **Sumber:** Google Drive (dataset internal)
* **Jumlah film:** 694
* **Jumlah pengguna:** 395

### Variabel / Fitur:

* **Film**: `movieId`, `title`, `genres`, `avg_rating`, `rating_variance`, `unique_users`
* **User**: struktur dictionary dengan `rating_count`, `rating_avg`, dan `movies` (film yang dirating user beserta nilai ratingnya)

### Exploratory Data Analysis (EDA):

* Genre unik: 14 genre termasuk Action, Drama, Comedy, dll.
* Rating rata-rata user: 3.74
* Distribusi rating: sebagian besar user memberikan rating antara 3.5 – 4.5
* Visualisasi:

  * Distribusi genre (bar chart dan pie chart)
  * Histogram distribusi rating user
  * Scatter plot antara jumlah film yang dirating dan rating rata-rata

---

## Data Preparation

### Langkah-langkah:

1. Mount Google Drive dan ekstraksi dataset dari file zip.
2. Baca file: `movie_list.csv`, `user_to_genre.pickle`, dan `user_train_header.txt`
3. Hitung statistik film:

   * Jumlah pengguna unik
   * Rata-rata rating
   * Variansi rating
4. Buat matrix user–item untuk Collaborative Filtering
5. Siapkan vektor fitur film berdasarkan genre dan statistik untuk Content-Based Filtering

### Alasan Data Preparation:

* Menyediakan representasi numerik untuk proses similarity
* Membersihkan data dan menghindari missing values
* Mengoptimalkan proses training model dengan input yang terstandarisasi

---

## Modeling and Results

### 1. Content-Based Filtering

* Representasi fitur film dibuat dari:

  * One-hot encoding genre
  * Statistik rating (avg, varians, pengguna unik)
* Similaritas dihitung menggunakan `cosine_similarity` antar film
* Untuk setiap user, rekomendasi diberikan berdasarkan film yang pernah disukai dan film yang paling mirip dengan film tersebut

### 2. Collaborative Filtering (SVD)

* Matrix interaksi user-item dibuat dalam format sparse matrix
* Menggunakan `scipy.sparse.linalg.svds` untuk dekomposisi
* Memperoleh 15 faktor laten untuk user dan film
* Prediksi rating dihitung kembali dan digunakan untuk membuat rekomendasi

### Output: Top-N Recommendation

* Rekomendasi top-10 film ditampilkan untuk user tertentu
* Hasil menunjukkan film dengan skor prediksi tertinggi
* Disediakan hasil dari kedua pendekatan untuk perbandingan

### Kelebihan & Kekurangan:

| Pendekatan          | Kelebihan                                           | Kekurangan                                         |
| ------------------- | --------------------------------------------------- | -------------------------------------------------- |
| Content-Based       | Tidak tergantung user lain, cocok untuk cold start  | Rekomendasi cenderung homogen dan terbatas         |
| Collaborative (SVD) | Menangkap pola komunitas dan preferensi tersembunyi | Butuh data cukup, tidak cocok untuk user/item baru |

---

## Evaluation

### Metrik Evaluasi:

* **Root Mean Squared Error (RMSE)**:
  $RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$

* **Mean Absolute Error (MAE)**:
  $MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$

### Hasil Evaluasi:

* Digunakan pada model Collaborative Filtering:

  * RMSE dan MAE digunakan untuk mengukur akurasi prediksi rating
  * Content-Based dievaluasi secara kualitatif berdasarkan relevansi dan genre

---

## Struktur Laporan

* Mengikuti struktur modular: Overview → Business → Data → Modeling → Evaluation
* Disertai visualisasi, cuplikan kode, dan penjelasan langkah-langkah
* Dapat dibaca dengan baik dalam format notebook dan markdown

---

## Kesimpulan

Proyek ini berhasil membangun sistem rekomendasi film berbasis **hybrid filtering** dengan dua pendekatan utama:

1. **Content-Based Filtering** yang fokus pada karakteristik film.
2. **Collaborative Filtering (SVD)** yang memanfaatkan interaksi antar user dan item.

Sistem memberikan **top-N rekomendasi** yang cukup akurat. Evaluasi menggunakan RMSE dan MAE menunjukkan hasil yang memadai. Visualisasi dan EDA memberikan wawasan terhadap struktur dan distribusi data. Sistem ini dapat diperluas ke pendekatan **hybrid dengan pembobotan**, serta integrasi data lebih besar seperti ulasan teks atau rating eksplisit.

---

## Rekomendasi Pengembangan

* Integrasi data dari ulasan pengguna menggunakan NLP dan TF-IDF
* Penyesuaian bobot antar model untuk hybrid yang lebih dinamis
* Penambahan metrik evaluasi berbasis ranking seperti Precision\@K
* Pengujian sistem dengan user feedback secara langsung
