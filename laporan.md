# Laporan Proyek Machine Learning - Sistem Rekomendasi Film

## Project Overview

### Latar Belakang

Industri hiburan digital telah mengalami transformasi yang signifikan dalam dekade terakhir. Platform streaming seperti Netflix, Disney+, dan Amazon Prime Video menghadapi tantangan besar dalam membantu pengguna menemukan konten yang relevan dari katalog yang berisi ribuan hingga jutaan film. Menurut penelitian McKinsey & Company (2016), sistem rekomendasi yang efektif dapat meningkatkan engagement pengguna hingga 75% dan berkontribusi pada 35% pendapatan perusahaan streaming.

Permasalahan "information overload" menjadi semakin kompleks ketika pengguna dihadapkan pada pilihan yang terlalu banyak, yang dapat menyebabkan "choice paralysis" dan menurunkan kepuasan pengguna. Riset dari Netflix menunjukkan bahwa pengguna rata-rata menghabiskan 18 menit untuk mencari konten sebelum akhirnya memutuskan untuk menonton sesuatu atau bahkan meninggalkan platform.

### Mengapa Proyek Ini Penting

1. **Peningkatan User Experience**: Sistem rekomendasi yang akurat dapat mengurangi waktu pencarian dan meningkatkan kepuasan pengguna
2. **Retensi Pengguna**: Rekomendasi yang personal dapat meningkatkan engagement dan mengurangi churn rate
3. **Business Value**: Sistem rekomendasi dapat meningkatkan watch time dan pada akhirnya revenue perusahaan

### Referensi

- Gomez-Uribe, C. A., & Hunt, N. (2015). The Netflix Recommender System: Algorithms, Business Value, and Innovation. ACM Transactions on Management Information Systems
- Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook. Springer

## Business Understanding

### Problem Statements

Berdasarkan analisis kebutuhan industri streaming, permasalahan utama yang ingin diselesaikan adalah:

1. **Bagaimana cara merekomendasikan film yang relevan berdasarkan preferensi genre pengguna?**
2. **Bagaimana sistem dapat mengidentifikasi film-film dengan karakteristik serupa untuk diversifikasi rekomendasi?**
3. **Bagaimana mengukur efektivitas sistem rekomendasi dalam memberikan rekomendasi yang personal dan beragam?**

### Goals

Tujuan dari proyek ini adalah:

1. **Membangun sistem rekomendasi content-based filtering** yang dapat memberikan rekomendasi film personal berdasarkan riwayat preferensi genre pengguna
2. **Mengimplementasikan fitur pencarian film serupa** berdasarkan kesamaan genre menggunakan cosine similarity
3. **Mengembangkan sistem evaluasi** yang dapat mengukur tingkat personalisasi dan coverage genre dari rekomendasi yang dihasilkan

### Solution Approach

Untuk mencapai goals yang telah ditetapkan, proyek ini menggunakan pendekatan **Content-Based Filtering** dengan alasan:

1. **Content-Based Filtering**: Memanfaatkan fitur-fitur intrinsik dari item (film) seperti genre untuk memberikan rekomendasi. Pendekatan ini cocok untuk mengatasi cold start problem dan memberikan rekomendasi yang dapat dijelaskan kepada pengguna.

2. **Cosine Similarity**: Digunakan untuk mengukur kesamaan antara profil pengguna dan fitur film, serta untuk mencari film-film yang serupa.

## Data Understanding

### Informasi Dataset

Dataset yang digunakan dalam proyek ini terdiri dari tiga komponen utama:

1. **Movie List**: 694 film dengan 3 atribut (movieId, title, genres)
2. **User Preferences**: Data preferensi 395 pengguna terhadap genre film
3. **User Header**: Metadata dengan 17 kolom untuk profil pengguna

**Sumber Data**: Dataset diperoleh dari file arsip yang berisi data MovieLens yang telah diproses.

### Variabel dan Fitur

#### 1. Movie Dataset (`content_movie_list.csv`)
- **movieId**: ID unik untuk setiap film (int64)
- **title**: Judul film beserta tahun rilis (object)
- **genres**: Genre film yang dipisahkan dengan "|" (object)

#### 2. User Preferences (`content_user_to_genre.pickle`)
Struktur data preferensi pengguna:
- **glist**: Array rating rata-rata pengguna untuk setiap genre
- **g_count**: Array jumlah film yang dinilai pengguna per genre
- **rating_count**: Total jumlah rating yang diberikan pengguna
- **rating_sum**: Total nilai rating yang diberikan pengguna

### Exploratory Data Analysis

#### Analisis Genre Film

Berdasarkan analisis distribusi genre, ditemukan bahwa:

```
Top 10 Genre Film:
1. Comedy (296 film) - 42.7%
2. Drama (281 film) - 40.5%
3. Action (234 film) - 33.7%
4. Thriller (211 film) - 30.4%
5. Adventure (166 film) - 23.9%
6. Romance (130 film) - 18.7%
7. Sci-Fi (127 film) - 18.3%
8. Crime (124 film) - 17.9%
9. Fantasy (88 film) - 12.7%
10. Mystery (59 film) - 8.5%
```

**Insight**: Comedy dan Drama mendominasi dataset, menunjukkan preferensi umum terhadap genre yang memiliki appeal luas.

#### Analisis Preferensi Pengguna

Dari analisis user preferences ditemukan bahwa:
- Semua pengguna memiliki preferensi terhadap tepat 6 genre
- Distribusi rating menunjukkan variasi yang signifikan antar pengguna
- Beberapa pengguna memiliki data rating yang terbatas

## Data Preparation

### 1. Data Cleaning

```python
# Remove rows with missing essential information
movie_clean = movie_list.dropna(subset=['title']).copy()
# Handle missing genres
movie_clean['genres'] = movie_clean['genres'].fillna('Unknown')
```

**Hasil**: Dataset tetap memiliki 694 film setelah cleaning (tidak ada missing values).

### 2. Genre Feature Engineering

```python
def preprocess_genres(genre_string):
    """Convert genre string to list of genres"""
    if pd.isna(genre_string) or genre_string == 'Unknown':
        return ['Unknown']
    return [g.strip() for g in str(genre_string).split('|')]

# Create binary genre matrix using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movie_clean['genre_list'])
```

**Hasil**: Berhasil membuat 14 fitur genre dalam bentuk binary matrix.

**Alasan**: Transformasi ini diperlukan untuk:
- Mengubah data kategorikal menjadi numerik
- Memungkinkan perhitungan similarity menggunakan cosine similarity
- Memfasilitasi perhitungan dot product antara user profile dan movie features

### 3. User Profile Creation

```python
def create_user_profile(user_id, user_preferences, genre_columns):
    profile = np.zeros(len(genre_columns))
    if user_id in user_preferences and 'glist' in user_preferences[user_id]:
        glist = user_preferences[user_id]['glist']
        if glist is not None and len(glist[0]) == len(genre_columns):
            profile = glist[0]
    return profile
```

**Tujuan**: Membuat vektor profil pengguna yang selaras dengan fitur genre film untuk perhitungan similarity.

## Modeling and Result

### Content-Based Filtering Implementation

Sistem rekomendasi diimplementasikan menggunakan class `ContentBasedRecommender` dengan komponen utama:

#### 1. Inisialisasi Model

```python
class ContentBasedRecommender:
    def __init__(self, movies_df, genre_matrix, user_profiles):
        self.movies_df = movies_df.reset_index(drop=True)
        self.genre_matrix = genre_matrix
        self.user_profiles = user_profiles
        self.similarity_matrix = cosine_similarity(self.genre_matrix)
```

#### 2. User-Based Recommendation

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

#### 3. Movie-to-Movie Similarity

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

### Top-N Recommendation Results

#### Contoh Rekomendasi untuk User 2:
```
Top 5 recommended movies:
1. Sherlock Holmes: A Game of Shadows (2011) | Score: 24.98
2. Jurassic World (2015) | Score: 21.02
3. Children of Men (2006) | Score: 21.02
4. Jumper (2008) | Score: 21.02
5. The Hunger Games (2012) | Score: 21.02
```

#### Contoh Rekomendasi untuk User 4:
```
Top 5 recommended movies:
1. Shrek (2001) | Score: 18.50
2. Click (2006) | Score: 17.75
3. Inside Out (2015) | Score: 17.75
4. Moana (2016) | Score: 14.50
5. The Lego Movie (2014) | Score: 14.50
```

### Kelebihan dan Kekurangan Pendekatan

#### Kelebihan Content-Based Filtering:
1. **Tidak memerlukan data pengguna lain**: Dapat memberikan rekomendasi berdasarkan profil individual
2. **Transparansi**: Rekomendasi dapat dijelaskan berdasarkan fitur-fitur yang serupa
3. **Mengatasi cold start problem**: Dapat merekomendasikan item baru selama fitur-fiturnya tersedia
4. **Konsistensi**: Profil pengguna yang stabil menghasilkan rekomendasi yang konsisten

#### Kekurangan:
1. **Limited diversity**: Cenderung merekomendasikan item yang terlalu mirip
2. **Feature limitation**: Kualitas rekomendasi terbatas pada kualitas ekstraksi fitur
3. **Overspecialization**: Sulit untuk merekomendasikan item di luar preferensi yang sudah ada
4. **Cold start untuk pengguna baru**: Memerlukan informasi awal tentang preferensi pengguna

## Evaluation

### Metrik Evaluasi

#### 1. Personalization Score

**Formula**:
```
Personalization Score = (Σ |intersection(user_genres, movie_genres)|) / (|user_genres| × total_movies)
```

**Cara Kerja**: Metrik ini mengukur seberapa baik rekomendasi sesuai dengan preferensi genre pengguna dengan menghitung rasio genre yang cocok antara preferensi pengguna dan film yang direkomendasikan.

#### 2. Genre Coverage

**Formula**:
```
Genre Coverage = |recommended_genres| / |available_genres|
```

**Cara Kerja**: Metrik ini mengukur keberagaman genre dalam rekomendasi dengan menghitung proporsi genre yang muncul dalam daftar rekomendasi terhadap total genre yang tersedia.

### Hasil Evaluasi

Berdasarkan evaluasi terhadap 10 pengguna sampel:

```
Average Personalization Score: 0.543
Average Genre Coverage: 0.693
Number of users evaluated: 10
```

#### Interpretasi Hasil:

1. **Personalization Score (0.543)**:
   - Nilai sedang yang menunjukkan sistem cukup berhasil memberikan rekomendasi sesuai preferensi pengguna
   - Masih ada ruang untuk peningkatan dalam hal personalisasi

2. **Genre Coverage (0.693)**:
   - Nilai baik yang menunjukkan rekomendasi mencakup 69.3% dari seluruh genre available
   - Menunjukkan sistem tidak terlalu narrow dalam memberikan rekomendasi

#### Distribusi Skor Evaluasi:

Berdasarkan histogram distribusi:
- **Personalization Score**: Mayoritas pengguna mendapat skor 0.5-0.6, menunjukkan konsistensi sistem dalam memberikan rekomendasi yang cukup personal
- **Genre Coverage**: Distribusi yang relatif merata antara 0.5-0.9, menunjukkan variasi yang baik dalam keberagaman genre

### Kesimpulan Evaluasi

Sistem rekomendasi yang dikembangkan menunjukkan performa yang memuaskan dengan:
- Kemampuan personalisasi yang cukup baik (54.3%)
- Keberagaman genre yang tinggi (69.3%)
- Konsistensi dalam memberikan rekomendasi kepada berbagai tipe pengguna

Sistem ini cocok untuk implementasi awal platform streaming yang mengutamakan transparansi rekomendasi dan dapat menangani cold start problem untuk item baru.
