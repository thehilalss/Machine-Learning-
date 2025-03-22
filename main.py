import os

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import boxcox, normaltest
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Kaggle veri setini indirdim
path = kagglehub.dataset_download("uom190346a/sleep-health-and-lifestyle-dataset")

# Dosya adını belirle
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError("CSV dosyası bulunamadı.")
df = pd.read_csv(os.path.join(path, csv_files[0]))

# Veri setini görüntüleyelim
print(df.head())

# Eksik değerleri kontrol edelim
print('Eksik değerler:\n', df.isnull().sum())

# Eksik değerleri ortalama ile dolduralım ve bu işlemi sadece sayısal verileri olan sütunlar için yapalım
df.fillna(df.select_dtypes(include=np.number).mean(), inplace=True)

# Kategorik sütunları sayısal değerlere dönüştürmelim
categorical_columns = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for category in categorical_columns:
    df[category] = le.fit_transform(df[category])

# Korelasyon analizi (Örneğin, 'Quality of Sleep' ile korelasyonu inceleyelim)
if 'Quality of Sleep' in df.columns:
    print('Korelasyon (Quality of Sleep ile):\n', df.corr()['Quality of Sleep'].sort_values(ascending=False))

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Korelasyon Matrisi')
    plt.show()

# belirttiğimiz özellikleri seçelim
selected_features = ['Sleep Duration', 'Stress Level', 'Physical Activity Level']
if not all(col in df.columns for col in selected_features):
    raise ValueError("Seçilen özellikler veri setinde bulunamadı.")

cdf = df[selected_features + ['Quality of Sleep']]

# Veri setini eğitelim ve test kümelerine ayıralım
train_df, test_df = train_test_split(cdf, test_size=0.2, random_state=42)

# Basit Doğrusal Regresyon (örneğin 'Sleep Duration' ile)
simple_regression = linear_model.LinearRegression()
train_x_simple = train_df[['Sleep Duration']].values
train_y_simple = train_df[['Quality of Sleep']].values
simple_regression.fit(train_x_simple, train_y_simple)

# Çoklu Doğrusal Regresyon
multiple_regression = linear_model.LinearRegression()
train_x_multiple = train_df[selected_features].values
train_y_multiple = train_df[['Quality of Sleep']].values
multiple_regression.fit(train_x_multiple, train_y_multiple)

# Modelleri test edelim
test_x_simple = test_df[['Sleep Duration']].values
test_y_simple = test_df[['Quality of Sleep']].values
test_prediction_simple = simple_regression.predict(test_x_simple)
r2_simple = r2_score(test_y_simple, test_prediction_simple)
print(f'Basit Doğrusal Regresyon R2 Skoru: {r2_simple:.2f}')

test_x_multiple = test_df[selected_features].values
test_y_multiple = test_df[['Quality of Sleep']].values
test_prediction_multiple = multiple_regression.predict(test_x_multiple)
r2_multiple = r2_score(test_y_multiple, test_prediction_multiple)
print(f'Çoklu Doğrusal Regresyon R2 Skoru: {r2_multiple:.2f}')

# Normalizasyon ve D'Agostino K^2 testi
scaler = StandardScaler()
normalized_quality = scaler.fit_transform(df[['Quality of Sleep']])
k2, p = normaltest(normalized_quality)
print(f'D\'Agostino K^2: {k2[0]:.2f}, p-değeri: {p[0]:.4f}')
if p[0] < 0.05:
    print('Normal dağılım hipotezi reddedildi.')
else:
    print('Normal dağılım hipotezi reddedilemedi.')

# Dönüşüm türleri (Box-Cox, Log, Karekök)
transformed_data = pd.DataFrame()
transformed_data['Original'] = df['Quality of Sleep']
transformed_data['Box-Cox'], _ = boxcox(df['Quality of Sleep'] + 0.01)  # Negatif veya sıfır olmamalı
transformed_data['Log'] = np.log1p(df['Quality of Sleep'])
transformed_data['Square Root'] = np.sqrt(df['Quality of Sleep'])

print('Dönüşüm Türleri:\n', transformed_data.head())
