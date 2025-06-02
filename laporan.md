# Laporan Proyek Machine Learning - Sistem Rekomendasi Film

## Domain Proyek

Industri hiburan digital, khususnya platform streaming film, mengalami pertumbuhan pesat dalam dekade terakhir. Dengan jutaan konten yang tersedia di platform seperti Netflix, Amazon Prime, dan Disney+, pengguna sering mengalami kesulitan dalam menemukan film yang sesuai dengan preferensi mereka. Fenomena ini dikenal sebagai "information overload" atau kelebihan informasi yang dapat menurunkan kepuasan pengguna dan engagement terhadap platform.

Sistem rekomendasi telah menjadi solusi krusial untuk mengatasi masalah ini. Menurut penelitian McKinsey (2016), sistem rekomendasi dapat meningkatkan penjualan hingga 35% untuk platform e-commerce seperti Amazon. Sementara itu, Netflix melaporkan bahwa 80% konten yang ditonton pengguna berasal dari sistem rekomendasi mereka.

Pentingnya pengembangan sistem rekomendasi film yang efektif tidak hanya terletak pada peningkatan user experience, tetapi juga pada dampak bisnis yang signifikan. Platform yang dapat memberikan rekomendasi personal dan akurat akan memiliki keunggulan kompetitif dalam mempertahankan pengguna dan meningkatkan revenue melalui peningkatan konsumsi konten.

**Referensi:**
- Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Springer.
- Gomez-Uribe, C. A., & Hunt, N. (2015). The Netflix Recommender System. *ACM Transactions on Management Information Systems*.
- McKinsey & Company. (2016). How retailers can keep up with consumers.
Berikut adalah perbaikan sesuai permintaan kamu: bagian **"📂 Sumber Dataset"** sudah dipindahkan ke bawah bagian **Referensi**, jadi susunannya sekarang menjadi seperti ini:


### 📂 Sumber Dataset

Dataset yang digunakan dalam proyek ini bersumber dari:

🔗 **[Kaggle - Content-Based Filtering Dataset by Pushpak Gote](https://www.kaggle.com/datasets/pushpakgote/content-based-filtering-main)**

Dataset ini merupakan modifikasi dari MovieLens yang telah disesuaikan untuk keperluan eksperimen sistem rekomendasi berbasis content dan collaborative filtering.

## Business Understanding

### Problem Statements

1. **Bagaimana cara membantu pengguna menemukan film yang sesuai dengan preferensi mereka dari ribuan pilihan yang tersedia?**
   - Pengguna menghadapi kesulitan dalam menavigasi katalog film yang sangat besar
   - Waktu yang dihabiskan untuk mencari film yang diinginkan terlalu lama
   - Pengalaman pengguna menjadi tidak optimal karena information overload

2. **Bagaimana cara memberikan rekomendasi yang personal dan akurat berdasarkan riwayat rating pengguna dan karakteristik film?**
   - Setiap pengguna memiliki preferensi unik yang perlu dipahami sistem
   - Rekomendasi generic tidak memberikan value yang maksimal
   - Diperlukan sistem yang dapat belajar dari perilaku dan preferensi individual

3. **Bagaimana cara mengatasi masalah cold start untuk pengguna baru yang belum memiliki riwayat rating?**
   - Pengguna baru tidak memiliki historical data untuk basis rekomendasi
   - Sistem perlu dapat memberikan rekomendasi yang relevan sejak awal
   - Diperlukan pendekatan alternatif untuk user acquisition dan retention

### Goals

1. **Mengembangkan sistem rekomendasi yang dapat memberikan top-N film recommendations yang relevan untuk setiap pengguna**
   - Sistem dapat mengidentifikasi film-film yang kemungkinan besar akan disukai pengguna
   - Rekomendasi yang diberikan personal dan akurat
   - Meningkatkan satisfaction rate pengguna terhadap rekomendasi

2. **Menciptakan model yang dapat memprediksi rating pengguna terhadap film dengan akurasi tinggi**
   - Model dapat memahami pola preferensi pengguna berdasarkan historical data
   - Prediksi rating memiliki error yang minimal
   - Sistem dapat memberikan confidence level untuk setiap rekomendasi

3. **Membangun sistem yang dapat memberikan rekomendasi untuk pengguna baru berdasarkan preferensi genre**
   - Mengatasi cold start problem dengan efektif
   - Memberikan onboarding experience yang baik untuk pengguna baru
   - Sistem dapat beradaptasi dengan cepat seiring bertambahnya data pengguna

### Solution Statements

1. **Content-Based Filtering Implementation**
   - Menggunakan karakteristik film seperti genre, tahun rilis, dan rating rata-rata
   - Membangun profile preferensi pengguna berdasarkan genre favorit
   - Cocok untuk mengatasi cold start problem dan memberikan rekomendasi yang dapat dijelaskan

2. **Neural Network Collaborative Filtering**
   - Implementasi Neural Collaborative Filtering dengan dual neural network architecture
   - Menggunakan embedding layers untuk menangkap hubungan kompleks antara user dan item
   - Memanfaatkan non-linear patterns untuk meningkatkan akurasi prediksi

Kedua solusi akan dievaluasi menggunakan metrics seperti Mean Squared Error (MSE) untuk mengukur akurasi prediksi rating, dan recommendation relevance untuk mengukur kualitas rekomendasi yang dihasilkan.

## Data Understanding

Dataset yang digunakan dalam proyek ini bersumber dari MovieLens, yang merupakan salah satu dataset standar untuk penelitian sistem rekomendasi. Dataset ini telah diproses dan disesuaikan untuk keperluan proyek content-based filtering.

**Informasi Dataset:**
- **Jumlah film**: 694 film
- **Jumlah pengguna**: 395 pengguna
- **Sumber data**: Dataset MovieLens (modified for content-based filtering)
- **Format**: CSV dan Pickle files
- **Skala rating**: 1-5 (integer values)

### Struktur Data

#### 1. Movie List (content_movie_list.csv)
Variabel-variabel pada dataset film adalah sebagai berikut:
- **movieId**: ID unik untuk setiap film (integer)
- **title**: Judul film beserta tahun rilis dalam format "Title (Year)"
- **genres**: Genre film yang dipisahkan dengan separator "|" (contoh: "Comedy|Romance")

#### 2. User-Genre Preferences (content_user_to_genre.pickle)
Dataset ini berisi informasi preferensi setiap pengguna dalam format dictionary dengan struktur:
- **glist**: Array berisi rating rata-rata user untuk setiap genre (14 genre categories)
- **g_count**: Array berisi jumlah film yang dinilai untuk setiap genre
- **rating_count**: Total jumlah rating yang telah diberikan user
- **rating_sum**: Total penjumlahan seluruh rating yang diberikan
- **movies**: Dictionary berisi movieId sebagai key dan rating yang diberikan sebagai value
- **rating_ave**: Rating rata-rata yang diberikan user secara keseluruhan

### Exploratory Data Analysis

Dari analisis yang dilakukan terhadap dataset, ditemukan beberapa insights penting:

1. **Distribusi Genre**: Comedy dan Drama merupakan genre paling populer dengan jumlah film terbanyak
2. **Rating Distribution**: Distribusi rating cenderung normal dengan rata-rata sekitar 3.5
3. **User Behavior**: Terdapat variasi signifikan dalam jumlah rating yang diberikan antar pengguna (dari puluhan hingga ratusan rating)
4. **Genre Preferences**: Setiap pengguna menunjukkan preferensi yang berbeda terhadap genre tertentu
5. **Data Quality**: Dataset dalam kondisi bersih tanpa missing values atau duplicate entries

## Data Preparation

Pada tahap data preparation, dilakukan beberapa teknik preprocessing yang essential untuk mempersiapkan data sebelum modeling.

### Teknik Data Preparation yang Diterapkan

#### 1. Perhitungan Rating Rata-rata Film
Dilakukan agregasi rating dari seluruh pengguna untuk mendapatkan karakteristik objektif setiap film:

```python
# Menghitung total rating untuk setiap film
for user in user_to_genre:
    user_movies = user_to_genre[user]['movies']
    for movie in user_movies:
        movie_list.loc[movie_list['movieId']==movie, 'total_rating_sum'] += rating
        movie_list.loc[movie_list['movieId']==movie, 'total_rating_count'] += 1

# Menghitung rata-rata
movie_list['avg_rating'] = movie_list['total_rating_sum'] / movie_list['total_rating_count']
```

#### 2. Genre Separation dan One-Hot Encoding
Proses pemisahan multiple genres dan konversi ke format numerik:

```python
# Memisahkan genre yang digabung dengan separator
categories = row[2].split('|')
for category in categories:
    data_list.append({'movieId':row[0], 'year':row[1][-5:-1], 'genre':category})

# One-Hot Encoding untuk genre
ohe_categories = OneHotEncoder(handle_unknown='ignore')
genre_encoded = ohe_categories.fit_transform(my_item_vec['genres'].values.reshape(-1,1))
```

#### 3. Feature Engineering
Pembentukan feature vectors untuk user dan item:
- **User features**: 14 dimensi representing preferensi untuk setiap genre
- **Item features**: 16 dimensi terdiri dari tahun, rating rata-rata, dan genre encoding
- **Target variable**: Rating yang diberikan user terhadap film

#### 4. Data Scaling dan Normalization
Standardisasi features dan normalisasi target variable:

```python
# Standardization untuk features
scaler_items = StandardScaler()
items_scaled = scaler_items.fit_transform(items)

scaler_users = StandardScaler()  
users_scaled = scaler_users.fit_transform(users)

# Min-Max scaling untuk target ke range [-1,1]
y_scaler = MinMaxScaler((-1,1))
y_train_norm = y_scaler.fit_transform(y_train)
```

#### 5. Train-Test Split
Data dibagi dengan proporsi 80:20 menggunakan random_state=1 untuk ensure reproducibility.

### Alasan Data Preparation

1. **Rating Averaging**: Diperlukan untuk mendapatkan ground truth karakteristik setiap film berdasarkan konsensus pengguna
2. **Genre Separation**: Memungkinkan model memahami setiap genre secara individual dan menangani multi-genre films
3. **One-Hot Encoding**: Konversi categorical data ke numerical format yang dapat diproses oleh neural network
4. **Feature Scaling**: Memastikan semua features memiliki skala yang sama untuk optimasi gradient descent yang lebih stabil
5. **Train-Test Split**: Untuk evaluasi model yang unbiased dan mengukur generalization capability

## Modeling

### Pendekatan Neural Collaborative Filtering

Model yang dikembangkan menggunakan arsitektur Neural Collaborative Filtering dengan pendekatan dual neural network yang dapat menangkap complex non-linear relationships antara user dan item.

#### Arsitektur Model

**1. User Neural Network**
```python
users_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dense(32, activation='linear'),
])
```

**2. Item Neural Network**
```python
items_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32, activation='linear'),
])
```

**3. Model Integration**
- Kedua neural network menghasilkan embedding vector berdimensi 32
- L2 normalization diterapkan pada kedua embedding untuk stabilitas
- Dot product computation untuk menghitung similarity score
- Output berupa predicted rating

#### Custom Layer Implementation
```python
class L2NormalizeLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.linalg.l2_normalize(inputs, axis=1)
```

### Training Configuration
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam dengan learning rate 0.01
- **Epochs**: 30
- **Batch Size**: 32 (default)
- **Validation**: 20% dari training data

### Kelebihan dan Kekurangan Model

#### Kelebihan Neural Network Approach:
1. **Non-linear Pattern Recognition**: Dapat menangkap hubungan kompleks yang tidak dapat ditangkap oleh model linear
2. **Personalization**: Menghasilkan embedding yang highly personalized untuk setiap user dan item
3. **Scalability**: Architecture dapat di-scale untuk dataset yang lebih besar
4. **Feature Learning**: Dapat belajar representasi optimal dari raw features
5. **Cold Start Handling**: Dapat memberikan rekomendasi untuk user baru berdasarkan genre preferences

#### Kekurangan:
1. **Black Box Nature**: Model sulit untuk diinterpretasi dan dijelaskan kepada end-user
2. **Computational Cost**: Membutuhkan resource komputasi yang significant untuk training dan inference
3. **Data Hungry**: Memerlukan amount data yang cukup untuk mencapai performance optimal
4. **Hyperparameter Sensitivity**: Performance sangat bergantung pada tuning hyperparameter yang tepat
5. **Overfitting Risk**: Tanpa regularization yang tepat, model dapat overfit pada training data

## Evaluation

### Metrik Evaluasi yang Digunakan

#### 1. Mean Squared Error (MSE)
MSE merupakan metrik utama yang digunakan untuk mengukur akurasi prediksi rating.

**Formula:**
```
MSE = (1/n) * Σ(y_actual - y_predicted)²
```

**Cara Kerja MSE:**
- Menghitung selisih (error) antara rating aktual dan rating prediksi
- Mengkuadratkan setiap error untuk memberikan penalti yang lebih besar pada error yang besar
- Mengambil rata-rata dari semua squared errors
- Hasil yang lebih kecil menunjukkan prediksi yang lebih akurat

**Keunggulan MSE:**
- Sensitif terhadap outliers, sehingga model terdorong untuk menghindari prediksi yang sangat salah
- Differentiable, cocok untuk gradient-based optimization
- Mudah diinterpretasi dalam konteks rating prediction

#### 2. Distance-based Similarity untuk Rekomendasi
Untuk sistem rekomendasi berbasis similarity, digunakan Euclidean distance pada embedding space:

**Formula:**
```
Distance = √(Σ(v1ᵢ - v2ᵢ)²)
```

Dimana v1 dan v2 adalah embedding vectors dari dua film yang dibandingkan.

### Hasil Evaluasi

#### 1. Performance Metrics
- **Training MSE (final)**: 0.0985
- **Test MSE**: 0.1063
- **Generalization Gap**: 0.0078 (sangat kecil)

#### 2. Interpretasi Hasil

**Model Performance Analysis:**
1. **Excellent Accuracy**: MSE < 0.11 menunjukkan model dapat memprediksi rating dengan akurasi yang sangat tinggi
2. **Good Generalization**: Gap yang kecil antara training dan test loss (0.0078) menunjukkan model tidak mengalami overfitting dan dapat generalize dengan baik pada unseen data
3. **Stable Training**: Kurva loss menunjukkan konvergensi yang smooth tanpa oscillation yang signifikan

**Recommendation Quality Assessment:**
- Sistem berhasil memberikan rekomendasi yang relevan untuk new users berdasarkan genre preferences
- Top-N recommendations menunjukkan konsistensi dengan preferensi yang dinyatakan
- Similarity-based recommendations memberikan hasil yang logis berdasarkan karakteristik film

#### 3. Validation Results

**For New User Recommendations:**
Contoh rekomendasi untuk pengguna baru dengan preferensi Comedy, Romance, dan Sci-Fi:
1. Definitely, Maybe (Comedy) - Predicted Rating: 4.90
2. Be Kind Rewind (Comedy) - Predicted Rating: 4.90  
3. Bolt (Comedy) - Predicted Rating: 4.90

**For Existing User (ID: 36):**
Model berhasil memprediksi rating existing user dengan akurasi tinggi, menunjukkan kemampuan dalam memahami individual user preferences.

### Kesimpulan Evaluasi

Berdasarkan hasil evaluasi yang komprehensif, model Neural Collaborative Filtering telah berhasil mencapai seluruh tujuan proyek:

1. **High Prediction Accuracy**: Dengan MSE 0.1063, model mampu memprediksi rating dengan error yang sangat kecil
2. **Excellent Generalization**: Model tidak overfitting dan dapat bekerja dengan baik pada data baru
3. **Effective Recommendation**: Sistem dapat memberikan rekomendasi yang relevan baik untuk new users maupun existing users
4. **Robust Architecture**: Model stabil dan siap untuk implementasi production

Model ini telah memenuhi semua problem statements yang dirumuskan di awal dan siap untuk diimplementasikan dalam sistem rekomendasi film dengan performance yang memuaskan.
