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
- **Sumber Data**: [MovieLens Dataset](https://www.kaggle.com/datasets/pushpakgote/content-based-filtering-main)

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
**![Alt text](path/to/image.png "Judul gambar")**
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

### Top-N Recommendation Results

#### Contoh Rekomendasi untuk User 2:
**Profil Pengguna**: Preferensi tinggi untuk Action (3.9), Animation (5.0), Adventure (4.0), Comedy (4.2), Crime (4.0), Drama (4.0), Mystery (3.0), Romance (4.0), Sci-Fi (4.25), Thriller (3.875)

| Rank | Film | Genres | Score |
|------|------|--------|-------|
| 1 | Sherlock Holmes: A Game of Shadows (2011) | Action\|Adventure\|Comedy\|Crime\|Mystery\|Thriller | 24.98 |
| 2 | Jurassic World (2015) | Action\|Adventure\|Drama\|Sci-Fi\|Thriller | 21.02 |
| 3 | Children of Men (2006) | Action\|Adventure\|Drama\|Sci-Fi\|Thriller | 21.02 |
| 4 | Jumper (2008) | Action\|Adventure\|Drama\|Sci-Fi\|Thriller | 21.02 |
| 5 | The Hunger Games (2012) | Action\|Adventure\|Drama\|Sci-Fi\|Thriller | 21.02 |

#### Contoh Rekomendasi untuk User 3:
**Profil Pengguna**: Preferensi rendah untuk Action (0.5), Crime (0.5), Sci-Fi (0.5), Thriller (0.5). Pengguna dengan data terbatas (hanya 1 rating dengan nilai 0.5)

| Rank | Film | Genres | Score |
|------|------|--------|-------|
| 1 | Jurassic World (2015) | Action\|Adventure\|Drama\|Sci-Fi\|Thriller | 2.00 |
| 2 | Source Code (2011) | Action\|Drama\|Mystery\|Sci-Fi\|Thriller | 2.00 |
| 3 | The Hunger Games (2012) | Action\|Adventure\|Drama\|Sci-Fi\|Thriller | 2.00 |
| 4 | Star Trek: Nemesis (2002) | Action\|Drama\|Sci-Fi\|Thriller | 2.00 |
| 5 | Rise of the Planet of the Apes (2011) | Action\|Drama\|Sci-Fi\|Thriller | 2.00 |

#### Contoh Rekomendasi untuk User 4:
**Profil Pengguna**: Preferensi tinggi untuk Animation (4.0), Children (4.0), Adventure (2.5), Comedy (4.0), Crime (3.25), Fantasy (4.0), Romance (4.0)

| Rank | Film | Genres | Score |
|------|------|--------|-------|
| 1 | Shrek (2001) | Adventure\|Animation\|Children\|Comedy\|Fantasy\|Romance | 18.50 |
| 2 | Click (2006) | Adventure\|Comedy\|Drama\|Fantasy\|Romance | 17.75 |
| 3 | Inside Out (2015) | Adventure\|Animation\|Children\|Comedy\|Drama\|Fantasy | 17.75 |
| 4 | Moana (2016) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 14.50 |
| 5 | The Lego Movie (2014) | Action\|Adventure\|Animation\|Children\|Comedy\|Fantasy | 14.50 |

### Analisis Hasil

1. **User 2**: Profil menunjukkan preferensi tinggi dan beragam untuk berbagai genre action, adventure, sci-fi, dan thriller. Sistem merekomendasikan film dengan kombinasi genre tersebut dengan skor relevansi tinggi (20-25).

2. **User 3**: Merupakan kasus **cold start problem** dengan data sangat terbatas (hanya 1 rating dengan nilai rendah 0.5). Meskipun demikian, sistem masih dapat memberikan rekomendasi berdasarkan genre yang memiliki preferensi minimal. Skor yang rendah (2.00) mencerminkan ketidakpastian sistem karena kurangnya data historis pengguna.

3. **User 4**: Profil menunjukkan preferensi kuat untuk konten family-friendly dengan genre animasi, komedi, dan fantasy. Rekomendasi yang dihasilkan sangat sesuai dengan preferensi ini, dengan skor relevansi yang tinggi (14-18) untuk film-film yang cocok untuk keluarga.

### Insight Tambahan

#### Perbandingan Skor Rekomendasi:
- **User 2** (data lengkap): Skor 20-25 (tinggi)
- **User 3** (data minimal): Skor 2.00 (rendah, menunjukkan ketidakpastian)
- **User 4** (data sedang): Skor 14-18 (sedang-tinggi)

#### Penanganan Cold Start:
User 3 menunjukkan bagaimana sistem menangani pengguna dengan data terbatas. Meskipun skor rendah, sistem tetap dapat memberikan rekomendasi yang konsisten berdasarkan preferensi genre yang tersedia.

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
1. **Implementasi sistem rekomendasi end-to-end** yang dapat menangani berbagai skenario pengguna
2. **Pengembangan metrik evaluasi yang komprehensif** untuk mengukur personalisasi dan diversitas
3. **Analisis mendalam terhadap performa sistem** dengan contoh kasus nyata dari 3 user yang berbeda
4. **Dokumentasi lengkap proses pengembangan** dari data understanding hingga evaluasi

### Rekomendasi Pengembangan Lanjutan:
1. **Hybrid Approach**: Kombinasi content-based dan collaborative filtering untuk meningkatkan akurasi
2. **Feature Enhancement**: Menambahkan fitur rating, sutradara, aktor, dan metadata lainnya
3. **Real-time Learning**: Implementasi online learning untuk adaptasi preferensi yang dinamis
4. **A/B Testing**: Evaluasi performa sistem dalam production environment dengan user feedback

### Kontribusi Terhadap Business Goals:

Sistem ini berhasil mencapai tujuan bisnis yang ditetapkan:
- **Personalisasi tinggi**: Skor 0.543 menunjukkan rekomendasi yang relevan dengan preferensi pengguna
- **Diversitas terjaga**: Coverage 0.693 memastikan pengguna tidak terjebak dalam filter bubble
- **Penanganan cold start**: Sistem dapat memberikan rekomendasi bahkan untuk pengguna dengan data minimal
- **Scalability**: Arsitektur yang efisien untuk handling dataset besar

Sistem ini dapat menjadi foundation yang solid untuk pengembangan platform streaming yang lebih sophisticated dan user-centric, dengan fokus pada peningkatan user experience dan engagement.
