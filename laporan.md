# Laporan Proyek Machine Learning - Sistem Rekomendasi Film

**Nama:** Roby  
**Kelas:** Machine Learning Terapan

---

## Project Overview

### Latar Belakang

Industri hiburan digital, khususnya platform streaming film, telah mengalami pertumbuhan yang sangat pesat dalam beberapa tahun terakhir. Dengan semakin bertambahnya konten yang tersedia, pengguna sering kali menghadapi kesulitan dalam menemukan film yang sesuai dengan preferensi mereka dari ribuan pilihan yang ada. Fenomena ini dikenal sebagai "information overload" atau kelebihan informasi.

Menurut Netflix, lebih dari 80% konten yang ditonton oleh pengguna berasal dari sistem rekomendasi mereka. Hal ini menunjukkan betapa pentingnya peran sistem rekomendasi dalam meningkatkan user engagement dan kepuasan pengguna. Sistem rekomendasi yang efektif tidak hanya membantu pengguna menemukan konten yang relevan, tetapi juga dapat meningkatkan waktu yang dihabiskan pengguna di platform, yang pada akhirnya berdampak positif pada revenue perusahaan.

### Mengapa Proyek Ini Penting?

1. **Meningkatkan User Experience**: Membantu pengguna menemukan film yang sesuai dengan preferensi mereka dengan lebih mudah dan cepat
2. **Mengurangi Churn Rate**: Pengguna yang mendapatkan rekomendasi yang relevan cenderung lebih setia menggunakan platform
3. **Optimalisasi Katalog**: Memastikan semua konten dalam katalog mendapat eksposur yang seimbang
4. **Personalisasi**: Memberikan pengalaman yang unik untuk setiap pengguna berdasarkan preferensi individual

### Referensi

Penelitian menunjukkan bahwa sistem rekomendasi dapat meningkatkan click-through rate hingga 2-3 kali lipat dibandingkan sistem tanpa personalisasi. Studi oleh McKinsey & Company menyebutkan bahwa 35% pembelian di Amazon dan 75% aktivitas menonton di Netflix berasal dari sistem rekomendasi.

---

## Business Understanding

### Problem Statements

Berdasarkan analisis kebutuhan industri streaming, terdapat beberapa permasalahan utama yang perlu diselesaikan:

1. **Bagaimana cara merekomendasikan film yang relevan berdasarkan preferensi genre pengguna?**
   - Pengguna memiliki preferensi genre yang berbeda-beda
   - Sistem harus mampu memahami pola preferensi individual

2. **Bagaimana cara mengatasi cold start problem untuk pengguna baru?**
   - Pengguna baru belum memiliki riwayat rating atau interaksi
   - Sistem perlu memberikan rekomendasi awal yang meaningful

3. **Bagaimana cara memastikan diversitas dalam rekomendasi?**
   - Menghindari filter bubble dimana pengguna hanya direkomendasikan film dengan genre yang sama
   - Memastikan eksplorasi genre baru yang mungkin disukai pengguna

### Goals

Tujuan dari proyek ini adalah:

1. **Membangun sistem rekomendasi film yang personal dan akurat**
   - Memberikan rekomendasi berdasarkan preferensi genre pengguna
   - Mencapai tingkat personalisasi yang tinggi (>0.5)

2. **Mengembangkan sistem yang dapat menangani berbagai skenario pengguna**
   - Pengguna dengan data historis lengkap
   - Pengguna dengan data terbatas

3. **Menciptakan sistem yang scalable dan efisien**
   - Dapat menangani dataset dengan ribuan film dan pengguna
   - Waktu komputasi yang reasonable untuk real-time recommendation

### Solution Approach

Untuk mencapai goals yang telah ditetapkan, proyek ini mengusulkan dua pendekatan sistem rekomendasi:

#### 1. Content-Based Filtering
**Kelebihan:**
- Tidak memerlukan data dari pengguna lain (mengatasi cold start problem)
- Dapat memberikan rekomendasi yang transparan dan dapat dijelaskan
- Tidak terpengaruh oleh sparsity data rating pengguna

**Kekurangan:**
- Terbatas pada fitur konten yang tersedia
- Sulit untuk memberikan rekomendasi yang serendipitous (kejutan positif)
- Cenderung menghasilkan rekomendasi yang homogen

**Implementasi:**
- Menggunakan genre sebagai fitur utama
- Menerapkan TF-IDF Vectorization atau MultiLabelBinarizer untuk encoding genre
- Menggunakan Cosine Similarity untuk mengukur kemiripan

#### 2. Collaborative Filtering (Konsep)
**Kelebihan:**
- Dapat menemukan pola tersembunyi dalam preferensi pengguna
- Mampu memberikan rekomendasi yang surprising dan diverse
- Tidak memerlukan pengetahuan mendalam tentang fitur item

**Kekurangan:**
- Mengalami cold start problem untuk pengguna dan item baru
- Memerlukan data rating yang cukup untuk performa optimal
- Komputasi lebih intensif untuk dataset besar

---

## Data Understanding

### Informasi Dataset

Dataset yang digunakan dalam proyek ini terdiri dari:

- **Jumlah Film**: 694 film
- **Jumlah Pengguna**: 395 pengguna  
- **Kondisi Data**: Dataset bersih tanpa missing values
- **Sumber Data**: [MovieLens Dataset](https://www.kaggle.com/datasets/pushpakgote/content-based-filtering-main) (archive.zip)

### Struktur Data

Dataset terdiri dari beberapa file:

1. **content_movie_list.csv** (694 baris × 3 kolom)
   - `movieId`: ID unik untuk setiap film (int64)
   - `title`: Judul film dengan tahun rilis (object)
   - `genres`: Genre film yang dipisahkan dengan karakter '|' (object)

2. **content_user_to_genre.pickle**
   - Berisi preferensi genre 395 pengguna dalam format dictionary
   - Setiap user memiliki array preferensi untuk 14 genre

3. **content_user_train_header.txt**
   - Header file dengan 17 kolom untuk data training user

### Variabel dan Fitur

#### Film Features:
- **movieId**: Identifier unik untuk setiap film
- **title**: Judul lengkap film beserta tahun rilis
- **genres**: Kombinasi genre yang dimiliki film (Drama|Romance, Action|Thriller, dll.)

#### User Features:
- **glist**: Array rating rata-rata pengguna untuk setiap genre
- **g_count**: Array jumlah film yang telah di-rating per genre
- **rating_count**: Total jumlah rating yang diberikan pengguna
- **rating_sum**: Total nilai rating yang diberikan pengguna

### Exploratory Data Analysis (EDA)

#### 1. Distribusi Genre Film

Analisis menunjukkan 10 genre paling populer:
1. **Comedy** (296 film) - Genre paling dominan
2. **Drama** (281 film) - Hampir setara dengan comedy
3. **Action** (234 film) - Genre aksi cukup populer
4. **Thriller** (211 film) - Film menegangkan diminati
5. **Adventure** (166 film) - Genre petualangan
6. **Romance** (130 film) - Genre romantis
7. **Sci-Fi** (127 film) - Fiksi ilmiah
8. **Crime** (124 film) - Genre kriminal
9. **Fantasy** (88 film) - Genre fantasi
10. **Mystery** (59 film) - Genre misteri

#### 2. Analisis Preferensi Pengguna

- Semua pengguna memiliki preferensi untuk tepat 6 genre
- Distribusi preferensi seragam (uniform distribution)
- Tidak ada variasi dalam jumlah genre yang disukai antar pengguna

#### 3. Insight Data

- Dataset tidak memiliki missing values
- Genre Comedy dan Drama mendominasi katalog film
- Setiap pengguna telah dikonfigurasi dengan profil genre yang konsisten
- Data siap untuk digunakan tanpa extensive cleaning

---

## Data Preparation

### 1. Data Cleaning

#### Pembersihan Data Film
```python
# Remove rows with missing essential information
movie_clean = movie_list.dropna(subset=['title']).copy()

# Handle missing genres  
movie_clean['genres'] = movie_clean['genres'].fillna('Unknown')
```

**Hasil**: Semua 694 film berhasil dipertahankan karena tidak ada missing values.

**Alasan**: Memastikan tidak ada data yang hilang yang dapat mempengaruhi kualitas rekomendasi.

### 2. Feature Engineering

#### Preprocessing Genre
```python
def preprocess_genres(genre_string):
    """Convert genre string to list of genres"""
    if pd.isna(genre_string) or genre_string == 'Unknown':
        return ['Unknown']
    return [g.strip() for g in str(genre_string).split('|')]

movie_clean['genre_list'] = movie_clean['genres'].apply(preprocess_genres)
```

**Alasan**: Mengubah format string genre menjadi list untuk memudahkan pemrosesan dan analisis.

#### Binary Genre Matrix
```python
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movie_clean['genre_list'])
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)
```

**Hasil**: Matrix binary 694 × 14 untuk representasi genre setiap film.

**Alasan**: 
- Mengubah data kategorikal menjadi numerical untuk komputasi similarity
- Setiap genre direpresentasikan sebagai dimensi terpisah (One-Hot Encoding)
- Memungkinkan perhitungan cosine similarity antar film

### 3. User Profile Creation

#### Pembuatan Profil Pengguna
```python
def create_user_profile(user_id, user_preferences, genre_columns):
    profile = np.zeros(len(genre_columns))
    if user_id in user_preferences and 'glist' in user_preferences[user_id]:
        glist = user_preferences[user_id]['glist']
        if glist is not None and len(glist[0]) == len(genre_columns):
            profile = glist[0]
    return profile

user_profiles = {
    user_id: create_user_profile(user_id, user_to_genre, mlb.classes_)
    for user_id in user_to_genre
}
```

**Alasan**:
- Menstandarisasi profil pengguna dalam format yang sama dengan genre matrix film
- Memungkinkan perhitungan similarity antara user preference dan film features
- Menangani kasus edge dimana data user tidak lengkap

### 4. Alasan Tahapan Data Preparation

1. **Konsistensi Format**: Memastikan semua data dalam format yang dapat diproses oleh algoritma
2. **Numerical Representation**: Mengubah data kategorikal menjadi numerical untuk komputasi mathematical
3. **Dimensionality Alignment**: Memastikan user profile dan movie features memiliki dimensi yang sama
4. **Scalability**: Struktur data yang optimal untuk performa komputasi

---

## Modeling and Result

### Sistem Rekomendasi Content-Based Filtering

#### Arsitektur Model

```python
class ContentBasedRecommender:
    def __init__(self, movies_df, genre_matrix, user_profiles):
        self.movies_df = movies_df.reset_index(drop=True)
        self.genre_matrix = genre_matrix
        self.user_profiles = user_profiles
        self.similarity_matrix = cosine_similarity(self.genre_matrix)
```

#### Komponen Utama:

1. **Similarity Matrix**: Menggunakan cosine similarity untuk mengukur kemiripan antar film
2. **User-Movie Scoring**: Menghitung relevansi film berdasarkan profil pengguna
3. **Recommendation Engine**: Menghasilkan top-N recommendations

#### Metode Rekomendasi

##### 1. User-Based Recommendations
```python
def recommend_movies(self, user_id, n_recommendations=10):
    scores = self.get_user_movie_scores(user_id)
    if scores is None:
        return pd.DataFrame()
    
    recommendations = self.movies_df.copy()
    recommendations['relevance_score'] = scores
    recommendations = recommendations[recommendations['relevance_score'] > 0]
    
    return recommendations.sort_values('relevance_score', ascending=False).head(n_recommendations)
```

**Cara Kerja**:
- Mengalikan genre matrix film dengan profil pengguna (dot product)
- Menghasilkan skor relevansi untuk setiap film
- Mengurutkan berdasarkan skor tertinggi

##### 2. Item-Based Recommendations  
```python
def get_similar_movies(self, movie_title, n_recommendations=5):
    idx = self.movies_df[self.movies_df['title'].str.contains(movie_title, case=False)].index
    if idx.empty:
        return pd.DataFrame()
    
    sim_scores = list(enumerate(self.similarity_matrix[idx[0]]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    top_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]
    result = self.movies_df.iloc[top_indices].copy()
    result['similarity_score'] = [sim_scores[i+1][1] for i in range(n_recommendations)]
    
    return result
```

**Cara Kerja**:
- Mencari film berdasarkan judul
- Menggunakan pre-computed similarity matrix
- Mengembalikan film dengan similarity score tertinggi

### Hasil Rekomendasi Detail

#### User 2 - Profile Analisis:
**Preferensi Genre (glist)**: [3.9, 5.0, 0.0, 0.0, 4.0, 4.2, 4.0, 4.0, 0.0, 3.0, 4.0, 0.0, 4.25, 3.875]

**Interpretasi Profil**:
- **Comedy** (5.0): Sangat tinggi - genre favorit utama
- **Sci-Fi** (4.25): Tinggi - preferensi kuat untuk fiksi ilmiah
- **Adventure** (4.2): Tinggi - menyukai film petualangan
- **Action**, **Crime**, **Drama**, **Thriller** (4.0): Konsisten menyukai genre aksi dan drama
- **Western** (3.9): Sedang - cukup menyukai film koboi
- **Mystery** (3.0): Rendah - kurang tertarik misteri

**Top 5 Rekomendasi untuk User 2**:
1. **Sherlock Holmes: A Game of Shadows (2011)** | Genres: Action|Adventure|Comedy|Crime|Mystery|Thriller | Score: 24.98
2. **Jurassic World (2015)** | Genres: Action|Adventure|Drama|Sci-Fi|Thriller | Score: 21.02
3. **Children of Men (2006)** | Genres: Action|Adventure|Drama|Sci-Fi|Thriller | Score: 21.02
4. **Jumper (2008)** | Genres: Action|Adventure|Drama|Sci-Fi|Thriller | Score: 21.02
5. **The Hunger Games (2012)** | Genres: Action|Adventure|Drama|Sci-Fi|Thriller | Score: 21.02

**Analisis Rekomendasi User 2**:
- Film Sherlock Holmes mendapat skor tertinggi (24.98) karena menggabungkan banyak genre favorit: Comedy (5.0), Action (4.0), Adventure (4.2), Crime (4.0), dan Thriller (4.0)
- Film-film lainnya konsisten mendapat skor 21.02 karena kombinasi Action+Adventure+Sci-Fi+Thriller yang sesuai dengan preferensi tinggi user

---

#### User 3 - Profile Analisis:
**Preferensi Genre (glist)**: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]

**Interpretasi Profil**:
- **Cold Start User**: Hanya memiliki 1 rating (0.5) - pengguna dengan data sangat terbatas
- Preferensi sangat rendah untuk semua genre (0.5)
- Menunjukkan kasus challenging untuk sistem rekomendasi

**Top 5 Rekomendasi untuk User 3**:
1. **Jurassic World (2015)** | Genres: Action|Adventure|Drama|Sci-Fi|Thriller | Score: 2.00
2. **Source Code (2011)** | Genres: Action|Drama|Mystery|Sci-Fi|Thriller | Score: 2.00
3. **The Hunger Games (2012)** | Genres: Action|Adventure|Drama|Sci-Fi|Thriller | Score: 2.00
4. **Star Trek: Nemesis (2002)** | Genres: Action|Drama|Sci-Fi|Thriller | Score: 2.00
5. **Rise of the Planet of the Apes (2011)** | Genres: Action|Drama|Sci-Fi|Thriller | Score: 2.00

**Analisis Rekomendasi User 3**:
- Sistem memberikan skor rendah (2.0) karena preferensi user yang minimal
- Rekomendasi fokus pada film dengan genre Action+Sci-Fi+Thriller yang memiliki appeal luas
- Menunjukkan strategi "popular items" untuk cold start users

---

#### User 4 - Profile Analisis:
**Preferensi Genre (glist)**: [0.0, 4.0, 0.0, 4.0, 2.5, 4.0, 0.0, 3.25, 4.0, 0.0, 0.0, 4.0, 0.0, 0.0]

**Interpretasi Profil**:
- **Animation** (4.0): Tinggi - menyukai film animasi
- **Children** (4.0): Tinggi - preferensi untuk film family-friendly
- **Fantasy** (4.0): Tinggi - menyukai genre fantasi
- **Romance** (4.0): Tinggi - menyukai cerita romantis
- **Drama** (3.25): Sedang-tinggi - cukup menyukai drama
- **Adventure** (2.5): Sedang - preferensi moderat untuk petualangan

**Top 5 Rekomendasi untuk User 4**:
1. **Shrek (2001)** | Genres: Adventure|Animation|Children|Comedy|Fantasy|Romance | Score: 18.50
2. **Click (2006)** | Genres: Adventure|Comedy|Drama|Fantasy|Romance | Score: 17.75
3. **Inside Out (2015)** | Genres: Adventure|Animation|Children|Comedy|Drama|Fantasy | Score: 17.75
4. **Moana (2016)** | Genres: Adventure|Animation|Children|Comedy|Fantasy | Score: 14.50
5. **The Lego Movie (2014)** | Genres: Action|Adventure|Animation|Children|Comedy|Fantasy | Score: 14.50

**Analisis Rekomendasi User 4**:
- Shrek mendapat skor tertinggi (18.50) karena perfect match dengan 6 genre favorit: Adventure, Animation, Children, Fantasy, Romance
- Rekomendasi sangat sesuai dengan profil family-oriented dan animation lover
- Semua film yang direkomendasikan cocok untuk audience yang menyukai konten family-friendly

### Perbandingan Pola Rekomendasi

1. **User 2 (Action/Sci-Fi Enthusiast)**: Mendapat rekomendasi film dewasa dengan genre aksi dan sci-fi
2. **User 3 (Cold Start)**: Mendapat rekomendasi film populer dengan appeal luas
3. **User 4 (Family/Animation Lover)**: Mendapat rekomendasi film animasi dan family-friendly

Hasil ini menunjukkan bahwa sistem berhasil memberikan rekomendasi yang **personal** dan **kontekstual** berdasarkan profil masing-masing pengguna.

### Kelebihan dan Kekurangan Pendekatan

#### Kelebihan Content-Based Filtering:
1. **Tidak memerlukan data pengguna lain** - Mengatasi cold start problem
2. **Transparent dan explainable** - Dapat dijelaskan mengapa film direkomendasikan
3. **Tidak terpengaruh sparsity** - Bekerja baik meski data rating sedikit
4. **User independence** - Rekomendasi tidak bias oleh pengguna populer

#### Kekurangan Content-Based Filtering:
1. **Limited by content analysis** - Hanya berdasarkan fitur yang tersedia
2. **Over-specialization** - Cenderung merekomendasikan item serupa
3. **Lack of serendipity** - Sulit memberikan surprise recommendation
4. **New user cold start** - Tetap memerlukan preferensi awal pengguna

---

## Evaluation

### Metrik Evaluasi

#### 1. Personalization Score

**Formula**:
```
Personalization Score = (∑ intersection(user_preferred_genres, movie_genres) / |user_preferred_genres|) / total_movies
```

**Cara Kerja**:
- Mengidentifikasi genre yang disukai pengguna (rating > 3.0)
- Menghitung kesesuaian genre antara preferensi pengguna dan film yang direkomendasikan
- Menghasilkan skor antara 0-1, dimana 1 berarti perfect match

**Interpretasi**:
- 0.0-0.3: Personalisasi rendah
- 0.3-0.6: Personalisasi sedang  
- 0.6-1.0: Personalisasi tinggi

#### 2. Genre Coverage

**Formula**:
```
Coverage = |recommended_genres| / |available_genres|
```

**Cara Kerja**:
- Menghitung jumlah genre unik dalam rekomendasi
- Membandingkan dengan total genre yang tersedia
- Menghasilkan rasio keberagaman genre

**Interpretasi**:
- 0.0-0.4: Coverage rendah (rekomendasi homogen)
- 0.4-0.7: Coverage sedang (cukup beragam)
- 0.7-1.0: Coverage tinggi (sangat beragam)

### Hasil Evaluasi

#### Performa Sistem:
- **Average Personalization Score**: 0.543
- **Average Genre Coverage**: 0.693  
- **Number of users evaluated**: 10

#### Analisis Hasil:

1. **Personalization Score (0.543)**:
   - Menunjukkan tingkat personalisasi sedang-tinggi
   - Sistem berhasil memberikan rekomendasi yang cukup sesuai dengan preferensi pengguna
   - Masih ada ruang untuk peningkatan dalam akurasi personalisasi

2. **Genre Coverage (0.693)**:
   - Menunjukkan keberagaman genre yang baik dalam rekomendasi
   - Sistem tidak terjebak dalam filter bubble
   - Pengguna mendapat eksposur terhadap berbagai genre

#### Distribusi Metrik:

Berdasarkan histogram evaluasi:
- **Personalization Score**: Sebagian besar pengguna mendapat skor 0.5-0.6
- **Genre Coverage**: Mayoritas rekomendasi mencakup 0.5-0.9 dari semua genre

### Konteks Evaluasi

Metrik yang dipilih sesuai dengan:

1. **Problem Statement**: Mengukur relevansi dan keberagaman rekomendasi
2. **Business Goals**: Memastikan personalisasi dan diversitas
3. **Data Context**: Sesuai dengan struktur data genre-based

### Kesimpulan Evaluasi

Sistem rekomendasi yang dikembangkan menunjukkan:
- **Performa yang memuaskan** dalam memberikan rekomendasi personal
- **Keseimbangan yang baik** antara personalisasi dan diversitas
- **Potensi peningkatan** melalui optimasi algoritma dan fitur tambahan

---

## Kesimpulan

Proyek sistem rekomendasi film menggunakan Content-Based Filtering telah berhasil dikembangkan dengan hasil yang memuaskan. Sistem mampu memberikan rekomendasi yang personal dengan tingkat akurasi sedang-tinggi (0.543) sambil mempertahankan keberagaman genre yang baik (0.693).

### Pencapaian Utama:
1. **Implementasi sistem rekomendasi end-to-end** yang dapat menangani berbagai profil pengguna
2. **Pengembangan metrik evaluasi yang komprehensif** untuk mengukur personalisasi dan diversitas
3. **Analisis mendalam terhadap performa sistem** dengan contoh konkret dari 3 tipe pengguna berbeda
4. **Dokumentasi lengkap proses pengembangan** dari data understanding hingga evaluasi

### Insight Penting:
1. **Sistem berhasil menangani cold start problem** - User 3 dengan data minimal tetap mendapat rekomendasi yang reasonable
2. **Personalisasi efektif** - User 2 dan User 4 mendapat rekomendasi yang sangat sesuai dengan preferensi masing-masing
3. **Keberagaman terjaga** - Genre coverage 69.3% menunjukkan sistem tidak terjebak dalam filter bubble
4. **Transparansi tinggi** - Setiap rekomendasi dapat dijelaskan berdasarkan kesesuaian genre

### Rekomendasi Pengembangan Lanjutan:
1. **Hybrid Approach**: Kombinasi content-based dan collaborative filtering untuk meningkatkan akurasi dan serendipity
2. **Feature Enhancement**: Menambahkan fitur rating, sutradara, aktor, tahun rilis untuk personalisasi yang lebih mendalam
3. **Real-time Learning**: Implementasi online learning untuk adaptasi preferensi pengguna secara dinamis
4. **A/B Testing**: Evaluasi performa sistem dalam production environment dengan user feedback nyata
5. **Scalability Optimization**: Implementasi caching dan indexing untuk performa yang lebih baik pada dataset besar

Sistem ini dapat menjadi foundation yang solid untuk pengembangan platform streaming yang lebih sophisticated dan user-centric, dengan kemampuan adaptasi terhadap berbagai skenario pengguna dan preferensi yang beragam.
