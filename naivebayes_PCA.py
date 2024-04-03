# Mengimport library yang diperlukan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    classification_report,
)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB


# 1 Input
#  Mengimport data_normalizedset
data = pd.read_csv("kidney-disease.csv")

# Menampilkan data awal
print("Data awal".center(75, "="))
print(data.head())
print("=" * 75)


# 2 Prepocessing
# Pengecekan missing value
print("Pengecekan missing value".center(75, "="))
print(data.isnull().sum())
print("=" * 75)

# Memeriksa duplikat data
print("Jumlah baris duplikat dalam dataset:", data.duplicated().sum())

# Penanganan Missing value (Menghapus baris yang mengandung nilai null)
data.dropna(inplace=True)

# Menampilkan data setelah menghapus missing value
print("Data setelah menghapus missing value".center(75, "="))
print(data.isnull().sum())
print("=" * 75)

# Menampilkan boxplot untuk setiap fitur dalam dataset sebelum menangani outlier
plt.figure(figsize=(15, 10))
data.boxplot(rot=45)
plt.title("Boxplot untuk Setiap Fitur")
plt.show()

# Deteksi outliers dengan Z-score
z_scores = np.abs(stats.zscore(data._get_numeric_data()))
threshold = 3
outliers = (z_scores > threshold).any(axis=1)

# Menampilkan data yang memiliki outlier
print("Data yang memiliki outlier:".center(75, "="))
print(data[outliers])
print("=" * 75)

# Penanganan outliers dengan IQR (Interquartile Range)
for column in data._get_numeric_data().columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data[column] = np.where(
        data[column] < lower_bound,
        lower_bound,
        np.where(data[column] > upper_bound, upper_bound, data[column]),
    )

# Menampilkan boxplot untuk setiap fitur dalam dataset setelah penanganan outlier
plt.figure(figsize=(15, 10))
data.boxplot(rot=45)
plt.title("Boxplot untuk Setiap Fitur Setelah Penanganan Outlier")
plt.show()

# Menampilkan data setelah menangani outlier
print("Data setelah menangani outlier".center(75, "="))
print(data.head())
print("=" * 75)

# 3 Tansformasi
# Normalisasi data menggunakan Min-Max Scaling
minmax_scaler = MinMaxScaler()
data_minmax = minmax_scaler.fit_transform(data._get_numeric_data())

# Konstruksi DataFrame dari hasil normalisasi
data_normalized = pd.DataFrame(data_minmax, columns=data._get_numeric_data().columns)

# Menampilkan hasil normalisasi Min-Max Scaling
print("Hasil Normalisasi Min-Max Scaling".center(75, "="))
print(data_normalized.head())
print("=" * 75)

# 4 Seleksi Fitur
# PCA
print("Seleksi Fitur menggunakan PCA".center(75, "="))
# grouping yang dibagi menjadi dua
X = data_normalized.iloc[:, 0:-1].values
y = data_normalized.iloc[:, -1].values
y = np.where(y > 0, 1, 0)

# Menghitung komponen utama (principal components)
pca = PCA()
pca.fit(X)

# Menampilkan variance explained ratio dari setiap komponen
print("Variance Explained Ratio:")
print(pca.explained_variance_ratio_)
print()

# Memilih jumlah komponen yang akan digunakan (misalnya, 8 komponen)
n_components = 8
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X)

# Plot hasil PCA
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Dataset")
plt.colorbar(label="Class")
plt.show()

# Menampilkan hasil seleksi fitur menggunakan PCA
print("Hasil Seleksi Fitur menggunakan PCA:")
print("Data Variable Setelah PCA:")
print(X_train_pca)
print("Dimensi Data Variable Setelah PCA:", X_train_pca.shape)
print("=" * 75)

# 5 Klasifikasi
# grouping yang dibagi menjadi dua
print("Grouping Variable".center(75, "="))
X = X_train_pca
y = data_normalized.iloc[:, -1].values
y = np.where(y > 0, 1, 0)
print("Data Variable".center(75, "="))
print(X)
print("Data Kelas".center(75, "="))
print(y)
print("=============================================================================")


# Pembagian training dan testing
print("Splitting Data 20-80".center(75, "="))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("Instance variable data training".center(75, "="))
print(X_train)
print("Instance kelas data training".center(75, "="))
print(y_train)
print("Instance variable data testing".center(75, "="))
print(X_test)
print("Instance kelas data testing".center(75, "="))
print(y_test)
print("=============================================================================")
print()

# Pemodelan naive bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Prediksi naive bayes
print("Instance prediksi naive bayes: ")
Y_pred = nb_model.predict(X_test)
print(Y_pred)
print("=============================================================================")
print()

# 6 Evaluasi
# Menghitung confusion matrix
cm = confusion_matrix(y_test, Y_pred)

print("CLASSIFICATION REPORT naive bayes".center(75, "="))
# Menghitung akurasi
accuracy = accuracy_score(y_test, Y_pred)
# Menghitung presisi
precision = precision_score(y_test, Y_pred)
# Menampilkan precision, recall, f1-score, dan support
print(classification_report(y_test, Y_pred))

# Menghitung sensitivity (true positive rate)
TN = cm[1][1] * 1.0
FN = cm[1][0] * 1.0
TP = cm[0][0] * 1.0
FP = cm[0][1] * 1.0
sens = TN / (TN + FP) * 100

# Menghitung specificity (true negative rate)
spec = TP / (TP + FN) * 100

print("Akurasi : ", accuracy * 100, "%")
print("Sensitivity : " + str(sens))
print("Specificity : " + str(spec))
print("============================================================")
print()

# Menampilkan confusion matrix
print("Confusion matrix for naive bayes\n", cm)
f, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt=".0f", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("============================================================")
print()
