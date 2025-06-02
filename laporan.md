# Laporan Proyek Machine Learning - Roby saidi -Sistem Rekomendasi Film

## Domain Proyek

### Latar Belakang

Industri hiburan digital, khususnya platform streaming film, mengalami pertumbuhan pesat dalam beberapa tahun terakhir. Dengan jutaan film yang tersedia, pengguna sering menghadapi kesulitan dalam menemukan konten yang sesuai dengan preferensi mereka. Fenomena ini dikenal sebagai "information overload" atau kelebihan informasi, di mana terlalu banyak pilihan justru membuat pengambilan keputusan menjadi sulit.

Sistem rekomendasi telah menjadi solusi krusial untuk mengatasi masalah ini. Netflix melaporkan bahwa 80% konten yang ditonton pengguna berasal dari sistem rekomendasi mereka, sementara YouTube menyatakan bahwa 70% waktu tonton berasal dari rekomendasi algoritma. Hal ini menunjukkan pentingnya sistem rekomendasi yang efektif dalam meningkatkan user engagement dan kepuasan pengguna.

### Mengapa Proyek Ini Penting

1. **Meningkatkan User Experience**: Membantu pengguna menemukan film yang relevan dengan cepat dan akurat
2. **Efisiensi Waktu**: Mengurangi waktu yang dihabiskan pengguna untuk mencari konten yang sesuai
3. **Personalisasi**: Memberikan pengalaman yang disesuaikan dengan preferensi individual
4. **Business Value**: Meningkatkan retention rate dan engagement pengguna pada platform streaming

### Referensi dan Riset Terkait

Berdasarkan penelitian Gomez-Uribe & Hunt (2016) dalam "The Netflix Recommender System", sistem rekomendasi yang efektif dapat meningkatkan user satisfaction hingga 20-30%. Studi lain oleh Ricci et al. (2015) dalam "Recommender Systems Handbook" menunjukkan bahwa content-based filtering sangat efektif untuk mengatasi cold start problem pada pengguna baru.
### Dataset
https://www.kaggle.com/datasets/pushpakgote/content-based-filtering-main
## Business Understanding

### Problem Statements

1. **Information Overload**: Pengguna kesulitan menemukan film yang sesuai preferensi dari ribuan pilihan yang tersedia
2. **Low Engagement**: Kurangnya personalisasi menyebabkan pengguna kurang tertarik untuk mengeksplorasi konten baru
3. **Cold Start Problem**: Sistem sulit memberikan rekomendasi yang relevan untuk pengguna baru dengan data historis terbatas

### Goals

1. **Membangun sistem rekomendasi content-based filtering** yang dapat memberikan rekomendasi film personal berdasarkan preferensi genre pengguna
2. **Menghasilkan top-N recommendations** yang relevan dan akurat untuk setiap pengguna
3. **Mencapai tingkat personalisasi yang baik** dengan coverage genre yang luas dalam rekomendasi

### Solution Approach

#### Content-Based Filtering
Pendekatan yang dipilih dalam proyek ini adalah **Content-Based Filtering** menggunakan analisis kemiripan genre. Metode ini bekerja dengan:
- Menganalisis karakteristik item (genre film)
- Membangun profil pengguna berdasarkan preferensi genre historis
- Menghitung similarity antara profil pengguna dengan karakteristik film
- Merekomendasikan film dengan similarity score tertinggi

**Kelebihan**:
- Tidak memerlukan data dari pengguna lain (mengatasi cold start problem)
- Transparansi dalam rekomendasi (mudah dijelaskan mengapa film direkomendasikan)
- Tidak terpengaruh oleh sparsity data

**Kekurangan**:
- Terbatas pada fitur yang telah didefinisikan
- Kurang mampu menemukan item yang unexpected tapi relevan
- Cenderung menghasilkan rekomendasi yang mirip (lack of diversity)

## Data Understanding

### Informasi Dataset

Dataset yang digunakan dalam proyek ini terdiri dari:
- **Jumlah film**: 694 film
- **Jumlah pengguna**: 395 pengguna
- **Kondisi data**: Bersih, tanpa missing values pada kolom utama
- **Format**: CSV dan Pickle files

**Sumber Data**: Dataset diambil dari arsip yang berisi informasi film dan preferensi pengguna untuk sistem rekomendasi content-based filtering.

### Variabel dan Fitur Data

#### 1. Movie Dataset (`content_movie_list.csv`)
- **movieId**: ID unik untuk setiap film (int64)
- **title**: Judul film lengkap dengan tahun rilis (object)
- **genres**: Daftar genre film yang dipisahkan dengan karakter '|' (object)

#### 2. User Preferences Data (`content_user_to_genre.pickle`)
Struktur data pengguna berisi:
- **glist**: Array rating rata-rata pengguna untuk setiap genre
- **g_count**: Array jumlah film yang telah dinilai per genre

#### 3. User Header (`content_user_train_header.txt`)
File yang berisi informasi header untuk 17 kolom fitur pengguna.

### Exploratory Data Analysis

#### Analisis Genre Film
Distribusi genre dalam dataset menunjukkan:
1. **Comedy** (296 film) - Genre paling dominan
2. **Drama** (281 film) - Genre kedua terpopuler
3. **Action** (234 film) - Genre ketiga
4. **Thriller** (211 film)
5. **Adventure** (166 film)

#### Analisis Preferensi Pengguna
- Setiap pengguna memiliki preferensi untuk tepat **6 genre**
- Distribusi preferensi seragam di semua pengguna
- Data preferensi tersedia untuk 395 pengguna unik

#### Visualisasi Data
Grafik distribusi genre menunjukkan bahwa film komedi dan drama mendominasi dataset, sementara genre seperti mystery dan fantasy lebih jarang ditemukan. Hal ini penting untuk dipertimbangkan dalam sistem rekomendasi agar tidak bias terhadap genre populer.

## Data Preparation

### Teknik Data Preparation yang Diterapkan

#### 1. Data Cleaning
```python
# Remove rows with missing essential information
movie_clean = movie_list.dropna(subset=['title']).copy()
# Handle missing genres
movie_clean['genres'] = movie_clean['genres'].fillna('Unknown')
```

**Alasan**: Memastikan tidak ada data kosong yang dapat mengganggu proses pemodelan.

#### 2. Genre Feature Engineering
```python
def preprocess_genres(genre_string):
    if pd.isna(genre_string) or genre_string == 'Unknown':
        return ['Unknown']
    return [g.strip() for g in str(genre_string).split('|')]

# Create binary genre matrix
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movie_clean['genre_list'])
```

**Alasan**:
- Mengubah format string genre menjadi representasi numerik
- Memungkinkan perhitungan similarity menggunakan cosine similarity
- Menciptakan 14 fitur genre unik dalam bentuk binary matrix

#### 3. User Profile Creation
```python
def create_user_profile(user_id, user_preferences, genre_columns):
    profile = np.zeros(len(genre_columns))
    if user_id in user_preferences and 'glist' in user_preferences[user_id]:
        glist = user_preferences[user_id]['glist']
        if glist is not None and len(glist[0]) == len(genre_columns):
            profile = glist[0]
    return profile
```

**Alasan**:
- Mengonversi preferensi pengguna menjadi vektor numerik
- Memungkinkan perhitungan dot product dengan matrix genre film
- Standardisasi format data untuk konsistensi perhitungan

#### Proses Data Preparation yang Diperlukan

1. **Genre Preprocessing**: Diperlukan untuk mengubah data tekstual menjadi format yang dapat diproses secara matematis
2. **Binary Encoding**: Diperlukan untuk representasi multi-label genre dalam format yang kompatible dengan algoritma similarity
3. **User Profile Vectorization**: Diperlukan untuk mengubah preferensi pengguna menjadi format yang dapat dibandingkan dengan karakteristik film

## Modeling and Result

### Arsitektur Sistem Rekomendasi

Sistem rekomendasi yang dibangun menggunakan kelas `ContentBasedRecommender` dengan komponen utama:

#### 1. Inisialisasi dan Similarity Matrix
```python
class ContentBasedRecommender:
    def __init__(self, movies_df, genre_matrix, user_profiles):
        self.movies_df = movies_df.reset_index(drop=True)
        self.genre_matrix = genre_matrix
        self.user_profiles = user_profiles
        self.similarity_matrix = cosine_similarity(self.genre_matrix)
```

#### 2. Perhitungan User-Movie Scores
```python
def get_user_movie_scores(self, user_id):
    user_profile = self.user_profiles[user_id]
    return np.dot(self.genre_matrix, user_profile)
```

#### 3. Generasi Rekomendasi
```python
def recommend_movies(self, user_id, n_recommendations=10):
    scores = self.get_user_movie_scores(user_id)
    recommendations = self.movies_df.copy()
    recommendations['relevance_score'] = scores
    return recommendations.sort_values('relevance_score', ascending=False)
```

### Hasil Top-N Recommendations

#### Contoh Rekomendasi untuk User 2:
1. **Sherlock Holmes: A Game of Shadows (2011)** - Score: 24.02
   - Genres: Action|Adventure|Comedy|Crime|Mystery|Thriller
2. **Jurassic World (2015)** - Score: 21.85
   - Genres: Action|Adventure|Drama|Sci-Fi|Thriller
3. **Children of Men (2006)** - Score: 21.02
   - Genres: Action|Adventure|Drama|Sci-Fi|Thriller

#### Contoh Rekomendasi untuk User 4:
1. **Shrek (2001)** - Score: 18.25
   - Genres: Adventure|Animation|Children|Comedy|Fantasy|Romance
2. **Click (2006)** - Score: 17.75
   - Genres: Adventure|Comedy|Drama|Fantasy|Romance
3. **Inside Out (2015)** - Score: 16.50
   - Genres: Adventure|Animation|Children|Comedy|Drama|Fantasy

### Movie-to-Movie Similarity

Sistem juga menyediakan fitur pencarian film serupa berdasarkan kemiripan genre:

```python
def get_similar_movies(self, movie_title, n_recommendations=5):
    # Find similar movies using cosine similarity matrix
    sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
```

### Kelebihan dan Kekurangan Pendekatan

#### Kelebihan Content-Based Filtering:
1. **Tidak ada Cold Start Problem**: Dapat memberikan rekomendasi untuk pengguna baru
2. **Transparency**: Rekomendasi mudah dijelaskan berdasarkan genre
3. **Independensi**: Tidak memerlukan data dari pengguna lain
4. **Konsistensi**: Rekomendasi stabil dan dapat diprediksi

#### Kekurangan:
1. **Limited Feature Space**: Hanya berdasarkan genre, mengabaikan faktor lain
2. **Overspecialization**: Cenderung merekomendasikan item yang terlalu mirip
3. **Lack of Serendipity**: Sulit menemukan item yang unexpected tapi relevan
4. **Static Recommendations**: Tidak adaptif terhadap perubahan preferensi temporal

## Evaluation

### Metrik Evaluasi yang Digunakan

#### 1. Personalization Score
**Formula**:
```
Personalization Score = (Matching Genres / Total User Preferred Genres) / Total Recommended Movies
```

**Cara Kerja**:
- Mengidentifikasi genre yang disukai pengguna (rating > 3.0)
- Menghitung berapa banyak genre yang cocok dalam setiap film rekomendasi
- Rata-rata skor kesesuaian untuk semua rekomendasi

#### 2. Genre Coverage
**Formula**:
```
Genre Coverage = Unique Genres in Recommendations / Total Available Genres
```

**Cara Kerja**:
- Mengumpulkan semua genre unik dalam daftar rekomendasi
- Menghitung proporsi terhadap total genre yang tersedia dalam dataset
- Mengukur diversity rekomendasi

### Hasil Evaluasi

Evaluasi dilakukan pada 10 pengguna sampel dengan hasil sebagai berikut:

| Metrik | Nilai | Interpretasi |
|--------|-------|--------------|
| **Average Personalization Score** | 0.543 | Rekomendasi cukup sesuai dengan preferensi pengguna (54.3%) |
| **Average Genre Coverage** | 0.693 | Rekomendasi mencakup 69.3% dari semua genre yang tersedia |
| **Number of Users Evaluated** | 10 | Evaluasi dilakukan pada sampel representatif |

### Analisis Hasil Berdasarkan Metrik

#### Personalization Score (0.543)
- **Interpretasi**: Sistem berhasil memberikan rekomendasi yang cukup personal dengan tingkat kesesuaian 54.3%
- **Signifikansi**: Nilai di atas 0.5 menunjukkan bahwa lebih dari setengah rekomendasi sesuai dengan preferensi pengguna
- **Ruang Perbaikan**: Masih ada potensi peningkatan hingga 45.7%

#### Genre Coverage (0.693)
- **Interpretasi**: Sistem berhasil memberikan diversity yang baik dengan mencakup hampir 70% dari semua genre
- **Signifikansi**: Menunjukkan bahwa rekomendasi tidak terlalu sempit atau bias pada genre tertentu
- **Keseimbangan**: Terdapat trade-off yang baik antara personalisasi dan diversity

### Distribusi Skor Evaluasi

Berdasarkan histogram yang dihasilkan:
- **Personalization Scores**: Terdistribusi normal dengan puncak di range 0.5-0.6
- **Coverage Scores**: Terdistribusi baik di range 0.5-0.9, menunjukkan konsistensi diversity

### Konteks dan Kesesuaian Metrik

Metrik evaluasi yang digunakan sesuai dengan:
- **Problem Statement**: Mengukur efektivitas personalisasi dan diversity
- **Business Goals**: Memastikan rekomendasi yang relevan dan beragam
- **Data Context**: Cocok untuk content-based filtering dengan data genre

### Kesimpulan Evaluasi

Sistem rekomendasi yang dibangun menunjukkan performa yang **cukup baik** dengan:
1. Tingkat personalisasi yang acceptable (54.3%)
2. Diversity genre yang baik (69.3%)
3. Konsistensi performa across different users

Hasil ini menunjukkan bahwa pendekatan content-based filtering berbasis genre berhasil menyelesaikan problem statement yang telah ditetapkan, meskipun masih ada ruang untuk perbaikan dalam hal akurasi personalisasi.
