# Abalone Yaş Grubu Sınıflandırması

Fiziksel ölçümlerden (boyut, ağırlık, cinsiyet) abalone deniz salyangozlarının yaş grubunu
**Young (≤8 halka) / Middle (9–11) / Old (≥12)** olarak tahmin eden, yedi denetimli öğrenme
algoritmasını karşılaştıran ve en iyi modellerden birini SHAP ile açıklayan bir proje.

📓 Tüm analiz: [`abalone_final.ipynb`](abalone_final.ipynb)

## İçerik

| Aşama | Yöntem |
|---|---|
| Veri | UCI / OpenML Abalone — 4177 gözlem, 4 hatalı kayıt temizlendi → 4173 |
| Keşifsel analiz | Tanımlayıcı istatistikler, dağılımlar, IQR ile aykırı gözlem incelemesi |
| Modeller | Lojistik Regresyon · KNN · Karar Ağacı · Budanmış Ağaç (cost-complexity) · Bagging · Random Forest · XGBoost |
| Ayarlama | `GridSearchCV`, 5 katlı Stratified CV, ölçüt F1-macro (yalnızca eğitim seti) |
| Dengesizlik | Yok / `class_weight` / SMOTENC — eğitim setinde CV ile karşılaştırma |
| Değerlendirme | Accuracy, macro Precision / Recall / Specificity / F1, OvR ROC-AUC, bootstrap güven aralığı |
| Açıklanabilirlik | SHAP — global önem, beeswarm, dependence, waterfall, decision plot |
| Ek | ipywidgets + Groq (Llama 3.1) ile etkileşimli tahmin aracı |

## Sonuçlar

Test seti (835 gözlem), `class_weight` stratejisiyle:

| Model | CV F1-macro (eğitim) | Test F1-macro | Test %95 GA | Test ROC-AUC |
|---|---|---|---|---|
| Bagging | 0.676 ± 0.023 | 0.645 | [0.613, 0.676] | 0.826 |
| Lojistik Regresyon | 0.676 ± 0.020 | 0.674 | [0.641, 0.703] | 0.838 |
| XGBoost | 0.675 ± 0.015 | **0.678** | [0.645, 0.708] | **0.844** |
| Random Forest | 0.675 ± 0.020 | 0.673 | [0.642, 0.703] | 0.835 |
| KNN | 0.650 ± 0.018 | 0.646 | [0.610, 0.676] | 0.821 |
| Budanmış Ağaç | 0.648 ± 0.013 | 0.616 | [0.584, 0.649] | 0.800 |
| Karar Ağacı | 0.637 ± 0.021 | 0.615 | [0.583, 0.647] | 0.806 |

![Model karşılaştırması](figures/model_karsilastirma.png)

- **XGBoost, Random Forest, Bagging ve Lojistik Regresyon istatistiksel olarak başabaştır**; güven aralıkları büyük ölçüde örtüşür.
  Tek karar ağaçları ve KNN daha zayıftır.
- En zor sınıf **Middle**'dır; hatalar çoğunlukla komşu sınıflar arasındadır (Young ↔ Old karışıklığı nadir).
- `class_weight` küçük bir iyileşme sağlamış, SMOTE katkı sağlamamıştır.

### SHAP (XGBoost)

![SHAP önem](figures/shap_importance.png)

- **Kabuk ağırlığı (ShellWeight)** en güçlü yaş göstergesidir: arttıkça *Old* skoru artar, *Young* skoru düşer.
- **Et ağırlığı (ShuckedWeight)** ters yönde çalışır: diğer ölçümler sabitken et ağırlığı yüksek olan birey daha genç tahmin edilir.

## Düzeltmeler

Projenin ilk sürümü incelenip aşağıdaki hatalar giderildi:

**Metodoloji**
1. **Test seti model seçiminde kullanılıyordu.** SMOTE vs. `class_weight` karşılaştırması test setinde yapılıp strateji
   test sonucuna göre seçiliyordu. Artık tüm seçimler eğitim setinde 5 katlı CV ile yapılıyor; test seti yalnızca bir kez, en sonda kullanılıyor.
2. **Ölçekleme sızıntısı.** `StandardScaler` tüm eğitim setine uydurulup ardından CV yapılıyordu. LR ve KNN için ölçekleyici artık
   `Pipeline` içinde, her katta yalnızca o katın eğitim kısmına uyduruluyor.
3. **SMOTE yalnızca eğitim katlarına uygulanıyor** (`imblearn.Pipeline`). Düz SMOTE, `Sex_F`/`Sex_M` 0/1 sütunlarında 0.37 gibi anlamsız
   değerler üretiyordu; bunun yerine bu sütunları kategorik kabul eden **SMOTENC** kullanıldı.
4. **"class_weight" senaryosu XGBoost ve KNN'de etkisizdi.** XGBoost'ta `class_weight` parametresi yok; artık eşdeğer dengeli
   `sample_weight` veriliyor (metadata routing ile yalnızca `fit`'e — aksi hâlde GridSearchCV ağırlıklı F1 hesaplıyordu).
   KNN'de sınıf ağırlığı kavramı olmadığı not edildi.
5. **Hiperparametreler elle kopyalanıyordu** (ör. `ccp_alpha=0.00294`, yuvarlanmış). Artık doğrudan `best_estimator_` kullanılıyor.
6. **Izgara sınırında kalan parametreler** (LR `C=10`, RF `max_depth=10`, XGBoost `learning_rate=0.1`) için ızgaralar genişletildi.
7. **"En iyi model" iddiası belirsizlik olmadan yapılıyordu.** CV std ve test F1 için bootstrap %95 güven aralıkları eklendi;
   ilk dört modelin farkının anlamlı olmadığı gösterildi.

**Yorumlar ve grafikler**

8. **Aykırı gözlem metni koda uymuyordu:** "3 satır çıkarıldı" yazıyordu; kod 4 satır çıkarıyordu (Height = 0.515 de dahil). Metin ve gerekçe
   (Height/Length oranı) düzeltildi.
9. **SHAP dependence yorumu yanlıştı:** "ShuckedWeight de aynı yönde hareket ediyor" — veriler tersini gösteriyor (Young için ρ = +0.88, Old için −0.88).
   Yorumlar artık sabit metin değil, hesaplanan bir yön tablosuna (Spearman ρ) dayanıyor.
10. **Decision plot hatalıydı:** her çizgi kendi özellik sırasına göre çizilip y-ekseni ortalama sıraya göre etiketleniyordu; çizgiler etiketlerle
    eşleşmiyordu. Ayrıca "Young'da ShellWeight sola itiyor (Old/Middle ihtimalini düşürüyor)" yorumu yanlıştı — ShellWeight Young skorunu sağa iter.
11. **SHAP birimi:** XGBoost çok sınıflı SHAP değerlerinin olasılık değil logit ölçeğinde olduğu belirtildi.
12. **Sınıf sırası:** `LabelEncoder` alfabetik sıralıyordu (Middle, Old, Young); hedef artık doğal sırasıyla kodlanıyor (Young < Middle < Old).

**Etkileşimli araç**

13. **🔐 Groq API anahtarı koda açık yazılmıştı.** Kaldırıldı; anahtar artık `GROQ_API_KEY` ortam değişkeninden okunuyor, yoksa araç LLM yorumu olmadan çalışıyor.
14. **Birim hatası:** girişler "mm" / "g" olarak etiketlenmişti ama model 1/200 ölçeğinde değer bekliyordu. Kullanıcı artık gerçek birimlerle giriyor, değerler ÷200 ile dönüştürülüyor.

**Temizlik:** tekrarlanan hücreler, kullanılmayan importlar ve kaldırılmış `multi_class` parametresi temizlendi.

## Çalıştırma

```bash
pip install -r requirements.txt
jupyter notebook abalone_final.ipynb
```

Etkileşimli araçta LLM yorumu için (isteğe bağlı):

```bash
export GROQ_API_KEY=...        # Windows: set GROQ_API_KEY=...
```

Veri seti ilk çalıştırmada OpenML'den otomatik indirilir. Grafikler `figures/`, eğitilmiş model `models/` klasörüne kaydedilir.

## Klasör yapısı

```
├── abalone_final.ipynb   # tüm analiz (çıktılarıyla)
├── figures/              # notebook'un ürettiği grafikler
├── requirements.txt
└── README.md
```
