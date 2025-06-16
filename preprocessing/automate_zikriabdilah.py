import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def preprocess_and_save_dataset(input_file, output_file):
    """
    Fungsi ini memuat data, membersihkan, memproses,
    dan menyimpan hasilnya ke file baru.
    """
    print(f"Membaca data dari: {input_file}")
    df = pd.read_csv("bestSelling_games_raw/bestSelling_games.csv")

    # Ubah tipe data tanggal
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # Hapus kolom yang tidak digunakan
    df_clean = df.drop(columns=['user_defined_tags', 'other_features', 'supported_languages'])

    # Isi nilai yang hilang (missing values)
    for col in ['rating', 'release_date']:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype == 'float64':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col] = df_clean[col].fillna(method='ffill')

    # Encode kolom kategorikal
    label_cols = ['developer', 'supported_os']
    le = LabelEncoder()
    for col in label_cols:
        df_clean[col] = le.fit_transform(df_clean[col])

    # === BAGIAN PENTING: Logika penyimpanan file ada DI DALAM FUNGSI ===
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Direktori dibuat: {output_dir}")

    df_clean.to_csv(output_file, index=False)
    print(f"✅ Dataset berhasil diproses dan disimpan di: {output_file}")

# Blok utama yang akan dijalankan
if __name__ == '__main__':
    preprocess_and_save_dataset(
        input_file='bestSelling_games_raw/bestSelling_games.csv',
        output_file='preprocessing/games_preprocessed.csv'
    )
