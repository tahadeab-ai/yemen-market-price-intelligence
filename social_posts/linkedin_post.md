# 🇾🇪 Forecasting Food & Fuel Market Dynamics in Conflict Economies: An Applied MLOps & Machine Learning Study

## 🌐 English Version

In protracted conflict environments, volatile currency devaluations and supply chain chokepoints directly threaten civilian food security. In Yemen, commodity price distributions exhibit extreme hyperinflationary skewness—with basket costs ranging from 24 YER to over 500,000 YER across 22 governorates.

I am excited to share an end-to-end Machine Learning and Spatial Intelligence repository: **"Yemen Food & Fuel Price Forecasting & Vulnerability Clustering"**, leveraging official United Nations World Food Programme (WFP) and Humanitarian Data Exchange (HDX) data spanning 2009 to 2026 (60,000+ verified price records).

### 📐 Applied Machine Learning Architecture & Key Engineering Decisions:

1. **Mitigating Hyperinflation Skew**:
   Applying `log1p(price)` stabilized severe right-skewness from **6.98 down to 1.07**, ensuring variance stability across severe economic shocks.
   
2. **Leakage-Free Temporal Split Protocol**:
   Unlike random k-fold splits that cause future price lookahead leakage, we implemented a strict chronological boundary (Train: 2009–2022 [75.1%], Out-of-Sample Test: 2023–2026 [24.9%]).
   
3. **Unsupervised Market Vulnerability Clustering**:
   Multi-evidence K-Means ($K=3$, Silhouette Score: **0.6629**) combined with 2D Principal Component Analysis (PCA explaining **96.14%** total variance) segmented regional markets into actionable vulnerability tiers—from low-volatility transit hubs to acute price shock frontiers (e.g., Aden City).
   
4. **Historical Price Shock Anomaly Detection**:
   An `IsolationForest` layer successfully detected 1,800 historical systemic price shock episodes, precisely aligning with documented currency depreciation waves and fuel import restrictions.
   
5. **Unified Scikit-Learn Pipelines & Experiment Matrix**:
   Composed leak-proof `ColumnTransformer` pipelines benchmarked across 3 diverse model families evaluated on the true Yemeni Rial scale (`expm1` inversion):
   * 🔹 **Ridge (L2 Linear Baseline)**: MAE = 5,832.81 YER | Log R² = 0.9286 (0.36s)
   * 🔹 **LightGBM (Gradient Boosting)**: MAE = 4,401.45 YER | Log R² = 0.9592 (1.59s)
   * 🏆 **Random Forest (Bagging Ensemble)**: MAE = 3,895.20 YER | Log R² = 0.9591

6. **Production Artifacts & Metadata Contract**:
   Automated serialization of models into Joblib with an audit-ready `metadata.json` capturing schemas, dependencies, and validation scores.

---

## 🇾🇪 النسخة العربية (Arabic Version)

يسرّني مشاركة هذا المشروع التقني المتقدم في تعلم الآلة وهندسة البيانات التطبيقية (MLOps): **"التنبؤ بديناميكيات أسعار الغذاء والوقود وعنقدة هشاشة الأسواق في اليمن"**، بالاعتماد على البيانات الرسمية لبرنامج الأغذية العالمي للأمم المتحدة (WFP) ومنصة تبادل البيانات الإنسانية (HDX) للفترة من 2009 إلى 2026 (+60,000 سجل تسعيري).

### 💡 أبرز الحلول والقرارات الهندسية في المشروع:
1. **معالجة تضخم وانقسام العملة**: تحويل الأسعار باستخدام `log1p` لخفض الالتواء الإحصائي من 6.98 إلى 1.07 وتفادي انفجار التدرج.
2. **بروتوكول التقسيم الزمني الصارم**: منع تسريب البيانات (Data Leakage) عبر تدريب النماذج على البيانات حتى نهاية 2022 واختبارها على بيانات 2023–2026.
3. **عنقدة هشاشة الأسواق (K-Means & PCA)**: تصنيف أسواق الـ 22 محافظة إلى 3 مستويات للهشاشة الاقتصادية بدقة سيوليت بلغت 0.6629 وشرح 96.14% من التباين عبر تحليل المكونات الرئيسية.
4. **اكتشاف صدمات الأسعار (Isolation Forest)**: عزل 1,800 صدمة سعرية مفاجئة تزامنت تاريخياً مع تقلبات صرف الريال وأزمات المشتقات النفطية.
5. **مصفوفة التجارب ونماذج التنبؤ**: حققت نماذج التعلم المجمّع دقة استثنائية (Log R² تجاوز 95.9%، ومتوسط خطأ مطلق MAE بلغ 3,895 ريال يمني).

🔗 **رابط المستودع والكود الكامل مفتوح المصدر**: متاح على GitHub و Google Colab.

#MachineLearning #MLOps #DataScience #DataLeakage #Yemen #FoodSecurity #LightGBM #ScikitLearn #Python #AIforGood
