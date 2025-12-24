import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler



import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys # Programı güvenle durdurmak için



GERCEK_GEN_DOSYASI = 'data.csv' 


try:
    labels_df = pd.read_csv('labels.csv', index_col=0)
    labels_df.index.name = 'Sample_ID'
    y = labels_df['Class'] 
    
    print(f"Etiketler başarıyla yüklendi. Toplam örnek sayısı: {len(y)}")
except FileNotFoundError:
    print("HATA: 'labels.csv' dosyası bulunamadı. Lütfen dosyanın klasörde olduğundan emin olun.")
    sys.exit() # Programı durdur


try:
    
    X_raw = pd.read_csv(GERCEK_GEN_DOSYASI, index_col=0)
    
    X = X_raw.loc[X_raw.index.intersection(y.index)]
    y = y.loc[X.index] # X ile y'yi senkronize etme
    
  
    X_numeric = X.select_dtypes(include=[np.number])

    print(f"Gen Verisi Yüklendi. Boyut: {X_numeric.shape[0]} Hasta, {X_numeric.shape[1]} Gen.")

except FileNotFoundError:
    print(f"HATA: '{GERCEK_GEN_DOSYASI}' adlı dosya bulunamadı. Lütfen dosya adını kontrol edin.")
    sys.exit()
except Exception as e:
    # Veri tipinden kaynaklanan diğer hatalar burada yakalanır
    print(f"HATA: Veri yüklenirken/işlenirken bir sorun oluştu. Detay: {e}")
    sys.exit()




X_train, X_test, y_train, y_test = train_test_split(
    X_numeric, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_numeric = pd.DataFrame(
    scaler.fit_transform(X_numeric),
    columns=X_numeric.columns,
    index=X_numeric.index
)



print("-" * 50)
print("VERİ BAŞARIYLA BÖLÜNDÜ:")
print("Eğitim Verisi Boyutu (X_train):", X_train.shape)
print("Test Verisi Boyutu (X_test):", X_test.shape)
print("-" * 50)



# PCA ile boyutu 50 bileşene indirge (t-SNE performansını artırmak için)
pca_50 = PCA(n_components=50, random_state=42)
X_pca = pca_50.fit_transform(X_numeric)
print(f"\nPCA (50 bileşen) ile boyutu {X_pca.shape[1]}'e indirildi.")

print("t-SNE uygulanıyor (Bu işlem büyük veride biraz zaman alabilir!)...")
start_time = time.time()
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30.0,
    max_iter=1000, 
    learning_rate=200.0 
)

X_tsne = tsne.fit_transform(X_pca)
end_time = time.time()

print(f"t-SNE tamamlandı. Süre: {end_time - start_time:.2f} saniye.")

tsne_df = pd.DataFrame(data = X_tsne, 
                     columns = ['tSNE-1', 'tSNE-2'],
                     index = X.index)
final_df_tsne = tsne_df.join(y)
final_df_tsne.columns = ['tSNE-1', 'tSNE-2', 'Kanser_Tipi']

plt.figure(figsize=(12, 10))
sns.scatterplot(x='tSNE-1', 
                y='tSNE-2', 
                hue='Kanser_Tipi',
                data=final_df_tsne, 
                palette='tab10',
                s=80,
                alpha=0.7)


plt.title('Visualization of Gene Clusters with t-SNE', fontsize=16)
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.legend(title='Cancer Type', loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
print("\n--- t-SNE Görselleştirmesi tamamlandı. ---")




print("\n--- Korelasyon Isı Haritası (Heatmap) ---")
N_GEN = 50 

try:
    top_n_genes = X_numeric.var().nlargest(N_GEN).index
    X_top_n = X_numeric[top_n_genes]
    print(f"En yüksek varyansa sahip ilk {N_GEN} gen seçildi. Korelasyon hesaplanıyor.")
except Exception as e:
    print(f"UYARI: Varyans hesaplanırken hata oluştu ({e}). İlk {N_GEN} sütunu kullanıyoruz.")
    X_top_n = X_numeric.iloc[:, :N_GEN]

correlation_matrix = X_top_n.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(
    correlation_matrix,
    annot=False, 
    cmap='vlag', 
    vmin=-1, vmax=1, 
    center=0, 
    linewidths=.5, 
    cbar_kws={'label': 'Correlation Coefficient'} # <--- ÇEVİRİ
)

plt.title(f'Correlation Map Among the Top {N_GEN} Most Variable Genes', fontsize=16)
plt.show()
print("\n--- Korelasyon Isı Haritası tamamlandı. ---")






logreg_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=5000)

print("Lojistik Regresyon modeli eğitiliyor...")
logreg_model.fit(X_train, y_train) 
print("Eğitim tamamlandı.")


y_pred_logreg = logreg_model.predict(X_test)


print("\n--- Model Değerlendirme Raporu (Lojistik Regresyon) ---")

print(classification_report(y_test, y_pred_logreg))

print("\nKarmaşıklık Matrisi (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred_logreg))

from sklearn.model_selection import cross_val_score
import numpy as np



print("\n" + "="*50)
print("LOJİSTİK REGRESYON ÇAPRAZ DOĞRULAMA SONUÇLARI (5-FOLD)")
print("="*50)


cv_scores = cross_val_score(logreg_model, X_numeric, y, cv=5, scoring='f1_weighted')

print(f"Tüm Çapraz Doğrulama F1 Skorları: {cv_scores}")
print(f"Ortalama F1 Skoru: {np.mean(cv_scores):.4f}")
print(f"Standart Sapma: {np.std(cv_scores):.4f}")

from sklearn.ensemble import RandomForestClassifier


print("\n" + "="*50)
print("RANDOM FOREST EĞİTİMİ VE DEĞERLENDİRMESİ")
print("="*50)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)

print("Random Forest modeli eğitiliyor...")
rf_model.fit(X_train, y_train)
print("Eğitim tamamlandı.")


y_pred_rf = rf_model.predict(X_test)

print("\n--- Model Değerlendirme Raporu (Random Forest) ---")
print(classification_report(y_test, y_pred_rf))

try:
    rf_roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test), multi_class='ovr')
    print(f"\nROC-AUC Skoru: {rf_roc_auc:.4f} (Multiclass / One-vs-Rest)")
except Exception as e:
    print(f"\nROC-AUC skoru hesaplanırken sorun oluştu: {e}")



print("\n" + "="*50)
print("RANDOM FOREST - EN ÖNEMLİ 10 GEN")
print("="*50)

feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)


top_10_features = feature_importances.nlargest(10)
print(top_10_features)


plt.figure(figsize=(10, 6))
top_10_features.sort_values(ascending=True).plot(kind='barh')
plt.title('Random Forest - Top 10 Most Important Genes (Feature Importance)', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Gene Name', fontsize=12)
plt.show()

print("\n--- Random Forest Değerlendirmesi tamamlandı. ---")

from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import shap
import warnings
warnings.filterwarnings('ignore') # Uyarıları gizlemek için

print("\n" + "="*50)
print("DESTEK VEKTÖR MAKİNELERİ (SVM) EĞİTİMİ VE DEĞERLENDİRMESİ")
print("="*50)


svm_model = SVC(kernel='rbf', random_state=42, class_weight='balanced', probability=True)

print("SVM modeli eğitiliyor... (Bu biraz zaman alabilir!)")
svm_model.fit(X_train, y_train) 
print("Eğitim tamamlandı.")

y_pred_svm = svm_model.predict(X_test)

print("\n--- Model Değerlendirme Raporu (SVM) ---")
print(classification_report(y_test, y_pred_svm))

try:
    svm_roc_auc = roc_auc_score(y_test, svm_model.predict_proba(X_test), multi_class='ovr')
    print(f"\nROC-AUC Skoru: {svm_roc_auc:.4f} (Multiclass / One-vs-Rest)")
except Exception as e:
    print(f"\nROC-AUC skoru hesaplanırken sorun oluştu: {e}")

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred_logreg)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=np.unique(y_test),
    yticklabels=np.unique(y_test)
)

plt.title('Confusion Matrix - Logistic Regression', fontsize=14)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()



print("\n" + "="*50)
print("TEMEL SİNİR AĞI (NN) EĞİTİMİ VE DEĞERLENDİRMESİ")
print("="*50)

# Keras, string sınıf etiketleriyle çalışmaz. Etiketleri kategorik formata dönüştür
y_train_encoded = pd.get_dummies(y_train).values
y_test_encoded = pd.get_dummies(y_test).values
n_classes = y_train_encoded.shape[1] 

nn_model = Sequential([
    # Giriş katmanı (Gen sayısı kadar nöron) ve gizli katman (512 nöron)
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)), 
    # İkinci gizli katman
    Dense(256, activation='relu'), 
    # Çıkış katmanı (Kanser tipi sayısı kadar nöron)
    Dense(n_classes, activation='softmax')
])

nn_model.compile(optimizer='adam', 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])

print("Sinir Ağı modeli eğitiliyor...")
history = nn_model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, verbose=0, validation_split=0.1)
print("Eğitim tamamlandı.")

y_pred_proba_nn = nn_model.predict(X_test, verbose=0)
y_pred_nn = np.argmax(y_pred_proba_nn, axis=1) # En yüksek olasılığa sahip sınıfı seç

target_names = pd.get_dummies(y_test).columns
y_test_decoded = np.argmax(y_test_encoded, axis=1)

print("\n--- Model Değerlendirme Raporu (Sinir Ağı) ---")
print(classification_report(y_test_decoded, y_pred_nn, target_names=target_names))



print("\n" + "="*50)
print("SHAP ANALİZİ: RANDOM FOREST MODEL KARARLARINI YORUMLAMA")
print("="*50)

class_names_list = ['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD']

explainer = shap.TreeExplainer(rf_model)

X_test_sample = X_test.sample(n=20, random_state=42)

print("SHAP değerleri hesaplanıyor...")
shap_values = explainer.shap_values(X_test_sample, check_additivity=False)

plt.figure(figsize=(12, 8))

if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    # Bazı SHAP sürümleri (samples, features, classes) şeklinde döner
    # Bunu summary_plot'un beklediği list formatına çekiyoruz
    shap_values_to_plot = [shap_values[:,:,i] for i in range(shap_values.shape[2])]
else:
    shap_values_to_plot = shap_values

shap.summary_plot(
    shap_values_to_plot, 
    X_test_sample, 
    feature_names=X_test_sample.columns,
    class_names=class_names_list,
    plot_type="bar",
    show=False
)

plt.title('SHAP Analysis: Global Gene Importance Across Cancer Types', fontsize=16)
plt.xlabel('Mean Absolute SHAP Value (Impact on Model)', fontsize=12)
plt.tight_layout()
plt.show()

for i, class_name in enumerate(class_names_list):
    plt.figure(figsize=(10, 6))
    
    # Doğru sınıfın değerlerini seçiyoruz
    current_val = shap_values_to_plot[i]
    
    shap.summary_plot(
        current_val,
        X_test_sample,
        feature_names=X_test_sample.columns,
        plot_type="bar",
        show=False
    )

    plt.title(f'Top Contributing Genes for: {class_name}', fontsize=14)
    plt.tight_layout()
    plt.show()


