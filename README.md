# Turkish Movie Review Sentiment Analysis

Türkçe film yorumlarının **pozitif / negatif** olarak sınıflandırılmasını amaçlayan uçtan uca bir doğal dil işleme (NLP) projesi. Veri web'den toplanmış, temizlenmiş, birden fazla özellik çıkarım yöntemiyle vektörleştirilmiş ve farklı makine öğrenmesi modelleriyle karşılaştırmalı olarak değerlendirilmiştir.

# Problem Tanımı

Amaç, kullanıcıların yazdığı film yorumlarının olumlu mu olumsuz mu olduğunu tahmin eden bir model geliştirmektir. Bu problem, doğal dil işleme alanında **duygu analizi (sentiment analysis)** kapsamında ikili sınıflandırma (binary classification) problemi olarak modellenmiştir.

# Pipeline (Çalışma Sırası)

Notebook'lar aşağıdaki sırayla çalıştırılmalıdır (her biri için `Kernel → Restart Kernel and Run All` önerilir):

# 1. Web Scraping — `web_scraping.ipynb`
`requests` ve `BeautifulSoup` kullanılarak çeşitli filmlerin (7. Koğuştaki Mucize, Ayla, The Dark Knight, Veda vb.) kullanıcı yorumları ve puanları web'den toplanmıştır.
Çıktı: `raw_reviews.csv`

# 2. Veri Ön İşleme — `Data_Prep.ipynb`
Ham veri üzerinde eksik değer temizliği, metin temizleme (stopword çıkarma, lemmatization — `nltk`) ve etiketleme yapılmıştır.
**Etiketleme stratejisi:** Orijinal puan skalasında nötr sayılabilecek (3–4 puan) yorumlar, sınıflandırma problemini net bir ikili ayrım (pozitif/negatif) olarak tutmak ve etiket gürültüsünü azaltmak amacıyla veri setinden çıkarılmıştır.
  Çıktı: `processed_reviews.csv`

# 3. Özellik Mühendisliği — `Feature_Eng.ipynb`
Temizlenmiş metinler üç farklı yöntemle vektörleştirilmiştir:
- **TF-IDF** (unigram + bigram, max 5000 özellik) + elle çıkarılan istatistiksel özellikler (soru işareti sayısı, büyük harf oranı, ortalama kelime uzunluğu, olumsuzluk/olumlu kelime sayısı vb.)
- **Bag-of-Words (BoW)** (unigram + bigram, max 5000 özellik)
- **Word2Vec** (Gensim, `vector_size=100`, skip-gram)

Çıktılar: `X_features.pkl`, `X_bow.pkl`, `X_w2v.npy`, `y_labels.pkl`, `tfidf_vectorizer.pkl`, `bow_vectorizer.pkl`, `word2vec.model`

# 4. Modelleme — `Models.ipynb`
Her özellik seti (TF-IDF+custom, BoW, Word2Vec) üzerinde aşağıdaki modeller eğitilip 5-fold stratified cross-validation ile karşılaştırılmıştır:
- Logistic Regression
- Linear SVM
- Random Forest
- MLP (Multi-Layer Perceptron)

En iyi performansı veren model `GridSearchCV` ile hiperparametre optimizasyonundan geçirilmiş ve `final_model.pkl` olarak kaydedilmiştir.

# 5. Analiz ve Değerlendirme — `Analysis.ipynb`
En iyi sonucu veren **TF-IDF + custom özellik seti** üzerinde detaylı analiz yapılmıştır:
- ROC eğrisi ve AUC skoru
- Confusion matrix
- Öğrenme eğrileri (learning curves)
- Yanlış sınıflandırılan örneklerin hata analizi

# Kullanılan Teknolojiler
- Dil: Python
- Veri toplama:requests, BeautifulSoup
- NLP: nltk, gensim (Word2Vec)
- Modelleme: scikit-learn (Logistic Regression, Linear SVM, Random Forest, MLP)
- Veri işleme: pandas, numpy
- Görselleştirme: matplotlib

#Kurulum ve Çalıştırma
bash
pip install -r requirements.txt
jupyter notebook

Ardından notebook'ları yukarıdaki sırayla (1 → 5) çalıştırın.

## Detaylı Rapor
Projenin tüm metodoloji ve sonuç detayları için: `FINALPROJECTREPORT.pdf`

---
Bu proje, Fatih Sultan Mehmet Vakıf Üniversitesi Yazılım Mühendisliği bölümü kapsamında geliştirilmiştir.Ekip olarak hazırlanmış bir projedir.
