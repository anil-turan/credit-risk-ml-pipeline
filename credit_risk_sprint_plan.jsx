import { useState } from "react";

const REPO_STRUCTURE = `credit-risk-ml-pipeline/
├── README.md
├── pyproject.toml
├── setup.py
├── requirements.txt
├── .github/
│   └── workflows/
│       └── ci.yml
├── .pre-commit-config.yaml
├── .gitignore
├── data/
│   ├── raw/              # Orijinal veri (git'e eklenmez)
│   └── processed/        # İşlenmiş veri
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_explainability.ipynb
│   └── 05_business_evaluation.ipynb
├── src/
│   └── credit_risk/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── preprocessor.py
│       ├── features/
│       │   ├── __init__.py
│       │   ├── engineer.py
│       │   └── selector.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   └── evaluator.py
│       └── explainability/
│           ├── __init__.py
│           └── shap_analysis.py
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── configs/
│   └── model_config.yaml
├── docs/
│   ├── model_card.md
│   └── governance_checklist.md
└── outputs/
    ├── figures/
    ├── models/
    └── reports/`;

const DAYS = [
  {
    day: 1,
    title: "Repo Setup & Veri Keşfi",
    date: "Pazartesi",
    hours: "3-4 saat",
    emoji: "🏗️",
    color: "#0f766e",
    objectives: [
      "GitHub repo oluştur, proje yapısını kur",
      "Veri setini indir ve ilk keşif yap",
      "EDA notebook'unu başlat"
    ],
    tasks: [
      {
        title: "1. GitHub repo oluştur",
        detail: "github.com → New Repository → 'credit-risk-ml-pipeline'. Public, MIT License, Python .gitignore. Clone et.",
        code: `git clone https://github.com/YOUR_USERNAME/credit-risk-ml-pipeline.git
cd credit-risk-ml-pipeline
mkdir -p data/{raw,processed} notebooks src/credit_risk/{data,features,models,explainability} tests configs docs outputs/{figures,models,reports}
touch src/credit_risk/__init__.py src/credit_risk/data/__init__.py
touch src/credit_risk/features/__init__.py src/credit_risk/models/__init__.py
touch src/credit_risk/explainability/__init__.py`,
      },
      {
        title: "2. Virtual environment & dependencies",
        detail: "Python 3.11+ kullan. Tüm bağımlılıkları requirements.txt'e yaz.",
        code: `python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# requirements.txt
cat > requirements.txt << 'EOF'
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
shap>=0.43.0
optuna>=3.4.0
imbalanced-learn>=0.11.0
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0
jupyter>=1.0.0
pytest>=7.4.0
pyyaml>=6.0.0
fairlearn>=0.9.0
pre-commit>=3.5.0
EOF

pip install -r requirements.txt`,
      },
      {
        title: "3. Veri setini indir",
        detail: "Kaggle API veya manuel indirme. Home Credit Default Risk veya LendingClub veri seti.",
        code: `# Kaggle API ile (önceden kaggle.json ayarla)
pip install kaggle
kaggle competitions download -c home-credit-default-risk -p data/raw/

# veya LendingClub
# kaggle datasets download -d wordsforthewise/lending-club -p data/raw/

# .gitignore'a ekle
echo "data/raw/" >> .gitignore
echo "data/processed/" >> .gitignore
echo "outputs/models/" >> .gitignore
echo "venv/" >> .gitignore`,
      },
      {
        title: "4. EDA Notebook başlat (01_eda.ipynb)",
        detail: "İlk hücrelerde veri yükleme, shape, dtypes, describe, null analizi.",
        code: `# 01_eda.ipynb - İlk hücreler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Veri yükleme
df = pd.read_csv('../data/raw/application_train.csv')

# Temel bilgiler
print(f"Shape: {df.shape}")
print(f"Target dağılımı:\\n{df['TARGET'].value_counts(normalize=True)}")
print(f"\\nEksik veri oranı (top 20):")
print((df.isnull().sum() / len(df) * 100).sort_values(ascending=False).head(20))
print(f"\\nVeri tipleri:\\n{df.dtypes.value_counts()}")`,
      },
    ],
    commit: "feat: initial project setup with repo structure and data loading",
    checklist: ["Repo oluşturuldu ve clone edildi", "Dizin yapısı kuruldu", "Dependencies yüklendi", "Veri seti indirildi", "İlk EDA hücreleri çalışıyor"]
  },
  {
    day: 2,
    title: "Derinlemesine EDA",
    date: "Salı",
    hours: "3-4 saat",
    emoji: "🔍",
    color: "#0f766e",
    objectives: [
      "Target distribution ve class imbalance analizi",
      "Numerik ve kategorik değişken dağılımları",
      "Korelasyon analizi ve ilk insight'lar"
    ],
    tasks: [
      {
        title: "1. Target analizi & class imbalance",
        detail: "Pozitif sınıf (default) oranını kontrol et. İmbalance stratejisi belirle.",
        code: `# Class imbalance görselleştirme
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Target dağılımı
df['TARGET'].value_counts().plot(kind='bar', ax=axes[0], color=['#0f766e', '#dc2626'])
axes[0].set_title('Target Distribution')
axes[0].set_xticklabels(['No Default (0)', 'Default (1)'], rotation=0)

# Yüzdesel
df['TARGET'].value_counts(normalize=True).plot(kind='pie', ax=axes[1], 
    autopct='%1.1f%%', colors=['#0f766e', '#dc2626'])
axes[1].set_title('Default Rate')

imbalance_ratio = df['TARGET'].value_counts()[0] / df['TARGET'].value_counts()[1]
print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
plt.tight_layout()
plt.savefig('../outputs/figures/target_distribution.png', dpi=150, bbox_inches='tight')`,
      },
      {
        title: "2. Numerik değişken dağılımları",
        detail: "Histogramlar, box plotlar. Outlier tespiti. Skewness kontrol.",
        code: `# Numerik değişkenler
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove('TARGET')
num_cols.remove('SK_ID_CURR')

# Skewness analizi
skewness = df[num_cols].skew().sort_values(ascending=False)
print("Yüksek skewness (>2):")
print(skewness[skewness.abs() > 2])

# Seçili değişkenler için dağılım
key_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
                'AMT_GOODS_PRICE', 'DAYS_BIRTH', 'DAYS_EMPLOYED']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, col in enumerate(key_features):
    ax = axes[i//3, i%3]
    df[col].hist(bins=50, ax=ax, color='#0f766e', alpha=0.7)
    ax.set_title(col)
    ax.axvline(df[col].median(), color='red', linestyle='--', label='median')
    ax.legend()
plt.tight_layout()
plt.savefig('../outputs/figures/numeric_distributions.png', dpi=150)`,
      },
      {
        title: "3. Kategorik değişkenler & target ile ilişki",
        detail: "Her kategorik değişkenin default rate'i ile ilişkisini incele.",
        code: `# Kategorik değişkenler ve target ilişkisi
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, col in enumerate(cat_cols[:6]):
    ax = axes[i//3, i%3]
    temp = df.groupby(col)['TARGET'].mean().sort_values(ascending=False)
    temp.plot(kind='bar', ax=ax, color='#7c3aed')
    ax.set_title(f'{col} vs Default Rate')
    ax.set_ylabel('Default Rate')
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('../outputs/figures/categorical_vs_target.png', dpi=150)`,
      },
      {
        title: "4. Korelasyon matrisi",
        detail: "Target ile en yüksek korelasyona sahip değişkenleri bul. Multicollinearity kontrol.",
        code: `# Target korelasyonları
target_corr = df[num_cols + ['TARGET']].corr()['TARGET'].drop('TARGET')
target_corr = target_corr.abs().sort_values(ascending=False)

print("TARGET ile en yüksek korelasyonlar (top 15):")
print(target_corr.head(15))

# Korelasyon heatmap (top 20 feature)
top_features = target_corr.head(20).index.tolist() + ['TARGET']
plt.figure(figsize=(14, 12))
sns.heatmap(df[top_features].corr(), annot=True, fmt='.2f', 
            cmap='RdBu_r', center=0, vmin=-1, vmax=1)
plt.title('Correlation Matrix - Top 20 Features')
plt.tight_layout()
plt.savefig('../outputs/figures/correlation_matrix.png', dpi=150)

# Yüksek korelasyonlu çiftler (multicollinearity)
corr_matrix = df[num_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [(col, idx, upper.loc[idx, col]) 
             for col in upper.columns for idx in upper.index 
             if upper.loc[idx, col] > 0.8]
print(f"\\nYüksek korelasyonlu çiftler (>0.8): {len(high_corr)}")
for c1, c2, corr in sorted(high_corr, key=lambda x: -x[2])[:10]:
    print(f"  {c1} <-> {c2}: {corr:.3f}")`,
      },
      {
        title: "5. Eksik veri haritası",
        detail: "Missing pattern analizi. MCAR/MAR tespiti.",
        code: `# Eksik veri haritası
missing_pct = (df.isnull().sum() / len(df) * 100)
missing_cols = missing_pct[missing_pct > 0].sort_values(ascending=False)

print(f"Eksik verili sütun sayısı: {len(missing_cols)} / {len(df.columns)}")
print(f"\\nEksik veri kategorileri:")
print(f"  >50% eksik: {(missing_pct > 50).sum()} sütun")
print(f"  20-50% eksik: {((missing_pct > 20) & (missing_pct <= 50)).sum()} sütun")
print(f"  <20% eksik: {((missing_pct > 0) & (missing_pct <= 20)).sum()} sütun")

# Görselleştirme
plt.figure(figsize=(12, 8))
missing_cols.head(30).plot(kind='barh', color='#dc2626')
plt.xlabel('Missing %')
plt.title('Top 30 Columns with Missing Data')
plt.tight_layout()
plt.savefig('../outputs/figures/missing_data_map.png', dpi=150)`,
      },
    ],
    commit: "feat: complete EDA with distributions, correlations, and missing data analysis",
    checklist: ["Target imbalance analizi tamam", "Numerik dağılımlar görselleştirildi", "Kategorik-target ilişkisi incelendi", "Korelasyon matrisi oluşturuldu", "Eksik veri haritası çıkarıldı", "outputs/figures/ klasörüne 5+ grafik kaydedildi"]
  },
  {
    day: 3,
    title: "Veri Temizleme & Preprocessing",
    date: "Çarşamba",
    hours: "3 saat",
    emoji: "🧹",
    color: "#0f766e",
    objectives: [
      "Eksik veri stratejisi uygula",
      "Outlier handling",
      "Data preprocessor modülünü yaz"
    ],
    tasks: [
      {
        title: "1. src/credit_risk/data/preprocessor.py",
        detail: "Modüler preprocessing pipeline. Reusable ve testable.",
        code: `# src/credit_risk/data/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer

class CreditRiskPreprocessor(BaseEstimator, TransformerMixin):
    """Credit risk data preprocessing pipeline."""
    
    def __init__(self, missing_threshold=0.5, outlier_method='iqr'):
        self.missing_threshold = missing_threshold
        self.outlier_method = outlier_method
        self.drop_cols_ = None
        self.num_imputer_ = None
        self.cat_imputer_ = None
        
    def fit(self, X, y=None):
        # Çok fazla eksik verili sütunları belirle
        missing_pct = X.isnull().sum() / len(X)
        self.drop_cols_ = missing_pct[missing_pct > self.missing_threshold].index.tolist()
        
        X_clean = X.drop(columns=self.drop_cols_)
        
        # Numerik ve kategorik ayrımı
        self.num_cols_ = X_clean.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = X_clean.select_dtypes(include=['object']).columns.tolist()
        
        # Imputer'ları fit et
        self.num_imputer_ = SimpleImputer(strategy='median')
        self.num_imputer_.fit(X_clean[self.num_cols_])
        
        self.cat_imputer_ = SimpleImputer(strategy='most_frequent')
        self.cat_imputer_.fit(X_clean[self.cat_cols_])
        
        # Outlier sınırları (IQR)
        if self.outlier_method == 'iqr':
            self.lower_ = {}
            self.upper_ = {}
            for col in self.num_cols_:
                Q1 = X_clean[col].quantile(0.01)
                Q3 = X_clean[col].quantile(0.99)
                self.lower_[col] = Q1
                self.upper_[col] = Q3
        
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.drop_cols_, errors='ignore')
        
        # Impute
        X[self.num_cols_] = self.num_imputer_.transform(X[self.num_cols_])
        X[self.cat_cols_] = self.cat_imputer_.transform(X[self.cat_cols_])
        
        # Outlier clipping (winsorize)
        if self.outlier_method == 'iqr':
            for col in self.num_cols_:
                X[col] = X[col].clip(self.lower_[col], self.upper_[col])
        
        return X`,
      },
      {
        title: "2. İlk unit test",
        detail: "Preprocessor'ın doğru çalıştığını doğrula.",
        code: `# tests/test_data.py
import pytest
import pandas as pd
import numpy as np
from src.credit_risk.data.preprocessor import CreditRiskPreprocessor

def test_preprocessor_removes_high_missing():
    df = pd.DataFrame({
        'a': [1, 2, np.nan, 4, 5],
        'b': [np.nan, np.nan, np.nan, np.nan, 5],  # 80% missing
        'c': ['x', 'y', 'x', 'y', 'x']
    })
    prep = CreditRiskPreprocessor(missing_threshold=0.5)
    prep.fit(df)
    result = prep.transform(df)
    assert 'b' not in result.columns
    assert 'a' in result.columns

def test_preprocessor_imputes_nulls():
    df = pd.DataFrame({
        'a': [1, 2, np.nan, 4, 5],
        'c': ['x', None, 'x', 'y', 'x']
    })
    prep = CreditRiskPreprocessor()
    result = prep.fit_transform(df)
    assert result.isnull().sum().sum() == 0`,
      },
    ],
    commit: "feat: add data preprocessor with imputation, outlier handling, and tests",
    checklist: ["preprocessor.py yazıldı", "Eksik veri imputation çalışıyor", "Outlier winsorization uygulandı", "Unit testler geçiyor", "pytest çalıştırıldı"]
  },
  {
    day: 4,
    title: "Feature Engineering",
    date: "Perşembe",
    hours: "3 saat",
    emoji: "⚙️",
    color: "#7c3aed",
    objectives: [
      "Domain-specific feature'lar oluştur (DTI, utilization)",
      "WoE/IV encoding uygula",
      "Feature engineering notebook'u tamamla"
    ],
    tasks: [
      {
        title: "1. src/credit_risk/features/engineer.py",
        detail: "Kredi riski domain bilgisine dayalı feature'lar.",
        code: `# src/credit_risk/features/engineer.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CreditRiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain-specific feature engineering for credit risk."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # === DOMAIN FEATURES ===
        
        # DTI Ratio (Debt-to-Income)
        X['DTI_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_INCOME_TOTAL'] + 1)
        
        # Credit-to-Income Ratio
        X['CREDIT_INCOME_RATIO'] = X['AMT_CREDIT'] / (X['AMT_INCOME_TOTAL'] + 1)
        
        # Credit-to-Goods Ratio (overpricing indicator)
        X['CREDIT_GOODS_RATIO'] = X['AMT_CREDIT'] / (X['AMT_GOODS_PRICE'] + 1)
        
        # Annuity-to-Credit Ratio (payment burden)
        X['ANNUITY_CREDIT_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_CREDIT'] + 1)
        
        # Age in years (DAYS_BIRTH is negative)
        X['AGE_YEARS'] = (-X['DAYS_BIRTH']) / 365.25
        
        # Employment years
        X['EMPLOYMENT_YEARS'] = (-X['DAYS_EMPLOYED']) / 365.25
        X['EMPLOYMENT_YEARS'] = X['EMPLOYMENT_YEARS'].clip(0, 50)
        
        # Employment-to-Age ratio (career stability)
        X['EMPLOYMENT_AGE_RATIO'] = X['EMPLOYMENT_YEARS'] / (X['AGE_YEARS'] + 1)
        
        # Income per family member
        X['INCOME_PER_FAMILY'] = X['AMT_INCOME_TOTAL'] / (X['CNT_FAM_MEMBERS'] + 1)
        
        # === INTERACTION FEATURES ===
        X['INCOME_CREDIT_INTERACTION'] = X['AMT_INCOME_TOTAL'] * X['AMT_CREDIT']
        
        # === BINNING ===
        X['AGE_GROUP'] = pd.cut(X['AGE_YEARS'], 
                                bins=[0, 25, 35, 45, 55, 65, 100],
                                labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
        
        return X`,
      },
      {
        title: "2. WoE/IV hesaplama",
        detail: "Weight of Evidence encoding — lojistik regresyon ile uyumlu, regulatory-friendly.",
        code: `# src/credit_risk/features/woe_encoder.py
import pandas as pd
import numpy as np

class WoEEncoder:
    """Weight of Evidence encoder with IV calculation."""
    
    def __init__(self, min_bin_size=0.05):
        self.min_bin_size = min_bin_size
        self.woe_maps_ = {}
        self.iv_scores_ = {}
    
    def fit(self, X, y, columns=None):
        if columns is None:
            columns = X.select_dtypes(include=['object', 'category']).columns
        
        for col in columns:
            woe_map, iv = self._calc_woe_iv(X[col], y)
            self.woe_maps_[col] = woe_map
            self.iv_scores_[col] = iv
        
        return self
    
    def _calc_woe_iv(self, feature, target):
        df = pd.DataFrame({'feature': feature, 'target': target})
        grouped = df.groupby('feature')['target'].agg(['sum', 'count'])
        grouped.columns = ['events', 'total']
        grouped['non_events'] = grouped['total'] - grouped['events']
        
        total_events = grouped['events'].sum()
        total_non_events = grouped['non_events'].sum()
        
        grouped['event_dist'] = grouped['events'] / (total_events + 1e-10)
        grouped['non_event_dist'] = grouped['non_events'] / (total_non_events + 1e-10)
        
        grouped['woe'] = np.log(
            (grouped['non_event_dist'] + 1e-10) / (grouped['event_dist'] + 1e-10)
        )
        grouped['iv'] = (grouped['non_event_dist'] - grouped['event_dist']) * grouped['woe']
        
        iv = grouped['iv'].sum()
        woe_map = grouped['woe'].to_dict()
        
        return woe_map, iv
    
    def transform(self, X):
        X = X.copy()
        for col, woe_map in self.woe_maps_.items():
            X[f'{col}_WOE'] = X[col].map(woe_map).fillna(0)
        return X
    
    def get_iv_summary(self):
        iv_df = pd.DataFrame({
            'Feature': self.iv_scores_.keys(),
            'IV': self.iv_scores_.values()
        }).sort_values('IV', ascending=False)
        iv_df['Predictive Power'] = iv_df['IV'].apply(
            lambda x: 'Useless' if x < 0.02 
            else 'Weak' if x < 0.1 
            else 'Medium' if x < 0.3 
            else 'Strong'
        )
        return iv_df`,
      },
      {
        title: "3. 02_feature_engineering.ipynb",
        detail: "Feature engineering notebook — tüm adımları belgele.",
        code: `# 02_feature_engineering.ipynb
from src.credit_risk.data.preprocessor import CreditRiskPreprocessor
from src.credit_risk.features.engineer import CreditRiskFeatureEngineer
from src.credit_risk.features.woe_encoder import WoEEncoder

# Pipeline uygula
preprocessor = CreditRiskPreprocessor()
df_clean = preprocessor.fit_transform(df)

engineer = CreditRiskFeatureEngineer()
df_feat = engineer.fit_transform(df_clean)

# WoE/IV analizi
woe = WoEEncoder()
cat_cols = df_feat.select_dtypes(include=['object', 'category']).columns
woe.fit(df_feat[cat_cols], df_feat['TARGET'], columns=cat_cols)

print("Information Value Summary:")
print(woe.get_iv_summary())

# Yeni feature'ların target ile korelasyonu
new_features = ['DTI_RATIO', 'CREDIT_INCOME_RATIO', 'CREDIT_GOODS_RATIO', 
                'ANNUITY_CREDIT_RATIO', 'EMPLOYMENT_AGE_RATIO', 'INCOME_PER_FAMILY']
for feat in new_features:
    corr = df_feat[feat].corr(df_feat['TARGET'])
    print(f"{feat}: correlation with TARGET = {corr:.4f}")`,
      },
    ],
    commit: "feat: domain-specific feature engineering with WoE/IV encoding",
    checklist: ["DTI, utilization ve diğer domain feature'lar oluşturuldu", "WoE/IV encoder yazıldı ve test edildi", "IV summary ile feature predictive power analizi", "02_feature_engineering.ipynb tamamlandı"]
  },
  {
    day: 5,
    title: "Feature Selection & Train/Test Split",
    date: "Cuma",
    hours: "2-3 saat",
    emoji: "🎯",
    color: "#7c3aed",
    objectives: [
      "Feature selection pipeline'ı kur",
      "Stratified train/test split",
      "Baseline model hazırlığı"
    ],
    tasks: [
      {
        title: "1. Feature selector modülü",
        detail: "Variance threshold + korelasyon filtresi + mutual information.",
        code: `# src/credit_risk/features/selector.py
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
import numpy as np
import pandas as pd

class FeatureSelector:
    def __init__(self, variance_threshold=0.01, corr_threshold=0.9):
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        
    def fit(self, X, y):
        # 1. Variance threshold
        vt = VarianceThreshold(threshold=self.variance_threshold)
        vt.fit(X.select_dtypes(include=[np.number]))
        self.low_var_cols_ = X.select_dtypes(include=[np.number]).columns[~vt.get_support()].tolist()
        
        # 2. High correlation pairs
        corr = X.select_dtypes(include=[np.number]).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.high_corr_cols_ = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
        
        # 3. Mutual information
        num_X = X.select_dtypes(include=[np.number]).fillna(0)
        mi = mutual_info_classif(num_X, y, random_state=42)
        self.mi_scores_ = pd.Series(mi, index=num_X.columns).sort_values(ascending=False)
        
        # Drop list
        self.drop_cols_ = list(set(self.low_var_cols_ + self.high_corr_cols_))
        
        return self
    
    def transform(self, X):
        return X.drop(columns=self.drop_cols_, errors='ignore')`,
      },
      {
        title: "2. Train/test split ve encoding",
        detail: "Stratified split + one-hot encoding + scaling pipeline.",
        code: `# Notebook 03_modeling.ipynb başlangıç
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Stratified split
X = df_feat.drop(columns=['TARGET', 'SK_ID_CURR'])
y = df_feat['TARGET']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train default rate: {y_train.mean():.4f}")
print(f"Test default rate: {y_test.mean():.4f}")

# Column transformer
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)`,
      },
    ],
    commit: "feat: feature selection pipeline and train/test split with encoding",
    checklist: ["Feature selector modülü yazıldı", "Low variance ve high correlation feature'lar belirlendi", "MI scores hesaplandı", "Stratified split yapıldı (%80/%20)", "ColumnTransformer pipeline hazır"]
  },
  {
    day: 6,
    title: "Model Eğitimi — Baseline + Advanced",
    date: "Cumartesi",
    hours: "4-5 saat",
    emoji: "🤖",
    color: "#dc2626",
    objectives: [
      "Logistic Regression baseline",
      "XGBoost + LightGBM eğitimi",
      "Optuna ile hyperparameter tuning"
    ],
    tasks: [
      {
        title: "1. Logistic Regression baseline",
        detail: "Regulatory-friendly baseline. Her zaman LR ile başla.",
        code: `from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# Pipeline: preprocess + model
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ))
])

lr_pipeline.fit(X_train, y_train)
y_pred_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
print(f"Logistic Regression AUC: {auc_lr:.4f}")`,
      },
      {
        title: "2. XGBoost + Optuna tuning",
        detail: "Bayesian hyperparameter optimization.",
        code: `import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'random_state': 42,
        'eval_metric': 'auc',
        'early_stopping_rounds': 50,
    }
    
    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_processed, y_train, 
                            cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

# Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)

print(f"Best AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Best model ile eğit
best_xgb = xgb.XGBClassifier(**study.best_params, random_state=42)
best_xgb.fit(X_train_processed, y_train)
y_pred_proba_xgb = best_xgb.predict_proba(X_test_processed)[:, 1]
print(f"XGBoost Test AUC: {roc_auc_score(y_test, y_pred_proba_xgb):.4f}")`,
      },
      {
        title: "3. LightGBM",
        detail: "Aynı Optuna yaklaşımıyla LightGBM.",
        code: `import lightgbm as lgb

def lgbm_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'is_unbalance': True,
        'random_state': 42,
        'verbosity': -1,
    }
    model = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_processed, y_train,
                            cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(lgbm_objective, n_trials=50, show_progress_bar=True)

best_lgbm = lgb.LGBMClassifier(**study_lgbm.best_params, random_state=42, verbosity=-1)
best_lgbm.fit(X_train_processed, y_train)
y_pred_proba_lgbm = best_lgbm.predict_proba(X_test_processed)[:, 1]
print(f"LightGBM Test AUC: {roc_auc_score(y_test, y_pred_proba_lgbm):.4f}")`,
      },
    ],
    commit: "feat: model training with LR baseline, XGBoost and LightGBM with Optuna tuning",
    checklist: ["LR baseline eğitildi ve AUC kaydedildi", "XGBoost Optuna ile tune edildi (50 trial)", "LightGBM Optuna ile tune edildi (50 trial)", "Tüm AUC skorları karşılaştırıldı", "En iyi model parametreleri kaydedildi"]
  },
  {
    day: 7,
    title: "Model Değerlendirme & Karşılaştırma",
    date: "Pazar (sabah)",
    hours: "2-3 saat",
    emoji: "📊",
    color: "#dc2626",
    objectives: [
      "Comprehensive evaluation: ROC, PR, KS, Lift",
      "Model karşılaştırma tablosu",
      "Confusion matrix analizi"
    ],
    tasks: [
      {
        title: "1. Model evaluator modülü",
        detail: "Tüm metrikleri hesaplayan reusable sınıf.",
        code: `# src/credit_risk/models/evaluator.py
from sklearn.metrics import (roc_auc_score, average_precision_score, 
    f1_score, precision_score, recall_score, confusion_matrix,
    roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import numpy as np

class ModelEvaluator:
    def __init__(self, y_true, y_proba, model_name="Model"):
        self.y_true = y_true
        self.y_proba = y_proba
        self.model_name = model_name
    
    def full_report(self, threshold=0.5):
        y_pred = (self.y_proba >= threshold).astype(int)
        metrics = {
            'ROC-AUC': roc_auc_score(self.y_true, self.y_proba),
            'PR-AUC': average_precision_score(self.y_true, self.y_proba),
            'F1': f1_score(self.y_true, y_pred),
            'Precision': precision_score(self.y_true, y_pred),
            'Recall': recall_score(self.y_true, y_pred),
            'KS Statistic': self._ks_statistic(),
            'Gini': 2 * roc_auc_score(self.y_true, self.y_proba) - 1,
        }
        return metrics
    
    def _ks_statistic(self):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba)
        return max(tpr - fpr)
    
    def plot_all(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        self._plot_roc(axes[0, 0])
        self._plot_pr(axes[0, 1])
        self._plot_ks(axes[1, 0])
        self._plot_score_dist(axes[1, 1])
        plt.suptitle(f'{self.model_name} — Evaluation Dashboard', fontsize=16)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    def _plot_roc(self, ax):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba)
        auc = roc_auc_score(self.y_true, self.y_proba)
        ax.plot(fpr, tpr, color='#0f766e', lw=2, label=f'AUC = {auc:.4f}')
        ax.plot([0,1], [0,1], 'k--', alpha=0.3)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title('ROC Curve'); ax.legend()
    
    def _plot_pr(self, ax):
        prec, rec, _ = precision_recall_curve(self.y_true, self.y_proba)
        ap = average_precision_score(self.y_true, self.y_proba)
        ax.plot(rec, prec, color='#7c3aed', lw=2, label=f'AP = {ap:.4f}')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title('PR Curve'); ax.legend()
    
    def _plot_ks(self, ax):
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_proba)
        ks = tpr - fpr
        ks_max_idx = np.argmax(ks)
        ax.plot(thresholds[1:], tpr[1:], label='TPR', color='#0f766e')
        ax.plot(thresholds[1:], fpr[1:], label='FPR', color='#dc2626')
        ax.axvline(thresholds[ks_max_idx+1], color='gray', ls='--', 
                   label=f'KS={ks[ks_max_idx]:.4f}')
        ax.set_xlabel('Threshold'); ax.set_title('KS Chart'); ax.legend()
    
    def _plot_score_dist(self, ax):
        ax.hist(self.y_proba[self.y_true==0], bins=50, alpha=0.5, 
                label='No Default', color='#0f766e', density=True)
        ax.hist(self.y_proba[self.y_true==1], bins=50, alpha=0.5, 
                label='Default', color='#dc2626', density=True)
        ax.set_xlabel('Predicted Probability'); ax.set_title('Score Distribution')
        ax.legend()`,
      },
      {
        title: "2. Model karşılaştırma tablosu",
        detail: "Tüm modelleri karşılaştır.",
        code: `# 03_modeling.ipynb — karşılaştırma
models = {
    'Logistic Regression': y_pred_proba_lr,
    'XGBoost': y_pred_proba_xgb,
    'LightGBM': y_pred_proba_lgbm,
}

results = []
for name, proba in models.items():
    evaluator = ModelEvaluator(y_test, proba, name)
    metrics = evaluator.full_report()
    metrics['Model'] = name
    results.append(metrics)
    evaluator.plot_all(f'../outputs/figures/{name.lower().replace(" ","_")}_evaluation.png')

comparison = pd.DataFrame(results).set_index('Model')
print(comparison.round(4))
comparison.to_csv('../outputs/reports/model_comparison.csv')`,
      },
    ],
    commit: "feat: comprehensive model evaluation with ROC, PR, KS curves and comparison",
    checklist: ["Evaluator modülü yazıldı", "ROC, PR, KS, Score Distribution grafikleri", "3 model karşılaştırma tablosu", "En iyi model belirlendi", "Grafikler outputs/figures/'a kaydedildi"]
  },
  {
    day: 8,
    title: "Explainability (SHAP)",
    date: "Pazar (öğleden sonra) + Pazartesi",
    hours: "3-4 saat",
    emoji: "🔮",
    color: "#d97706",
    objectives: [
      "SHAP analizi (global + local)",
      "Fairness analizi",
      "04_explainability.ipynb tamamla"
    ],
    tasks: [
      {
        title: "1. SHAP analizi",
        detail: "Summary, dependence, force ve waterfall plotlar.",
        code: `# 04_explainability.ipynb
import shap

# TreeExplainer (XGBoost/LightGBM için hızlı)
explainer = shap.TreeExplainer(best_xgb)  # veya best_lgbm
shap_values = explainer.shap_values(X_test_processed)

# Global: Summary plot (beeswarm)
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test_processed, 
                  feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('../outputs/figures/shap_summary.png', dpi=150, bbox_inches='tight')

# Global: Feature importance bar
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_processed, 
                  feature_names=feature_names, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('../outputs/figures/shap_importance.png', dpi=150, bbox_inches='tight')

# Dependence plots (top features)
for feat in ['DTI_RATIO', 'CREDIT_INCOME_RATIO', 'AGE_YEARS']:
    fig, ax = plt.subplots(figsize=(8, 5))
    feat_idx = feature_names.index(feat)
    shap.dependence_plot(feat_idx, shap_values, X_test_processed,
                        feature_names=feature_names, ax=ax, show=False)
    plt.savefig(f'../outputs/figures/shap_dep_{feat.lower()}.png', dpi=150)

# Local: Tek bir tahmin açıklama (waterfall)
idx = 0  # İlk test örneği
shap.plots.waterfall(shap.Explanation(
    values=shap_values[idx],
    base_values=explainer.expected_value,
    data=X_test_processed.iloc[idx],
    feature_names=feature_names
))
plt.savefig('../outputs/figures/shap_waterfall_example.png', dpi=150, bbox_inches='tight')`,
      },
      {
        title: "2. Fairness analizi (Fairlearn)",
        detail: "Demographic parity ve equalized odds kontrolü.",
        code: `# Fairness — Fairlearn
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from sklearn.metrics import accuracy_score

# Cinsiyet bazlı fairness
sensitive = X_test['CODE_GENDER']  # veya başka protected attribute
y_pred_best = (y_pred_proba_lgbm >= 0.5).astype(int)

mf = MetricFrame(
    metrics={'accuracy': accuracy_score, 'selection_rate': lambda y,p: p.mean()},
    y_true=y_test, y_pred=y_pred_best,
    sensitive_features=sensitive
)
print("Group-level metrics:")
print(mf.by_group)

dpd = demographic_parity_difference(y_test, y_pred_best, sensitive_features=sensitive)
print(f"\\nDemographic Parity Difference: {dpd:.4f}")
print(f"Disparate Impact Ratio: {mf.by_group['selection_rate'].min() / mf.by_group['selection_rate'].max():.4f}")`,
      },
    ],
    commit: "feat: SHAP explainability and fairness analysis with Fairlearn",
    checklist: ["SHAP summary plot (global importance)", "SHAP dependence plots (top 3 feature)", "SHAP waterfall (local explanation)", "Fairness analizi (demographic parity)", "04_explainability.ipynb tamamlandı"]
  },
  {
    day: 9,
    title: "Business Evaluation & Model Governance",
    date: "Salı",
    hours: "3 saat",
    emoji: "💼",
    color: "#d97706",
    objectives: [
      "Profit curve analizi (business metric)",
      "Model governance checklist (regulatory)",
      "Model card oluştur"
    ],
    tasks: [
      {
        title: "1. Profit curve & business impact",
        detail: "Modelin iş değerini parasal olarak göster.",
        code: `# 05_business_evaluation.ipynb
def profit_curve(y_true, y_proba, cost_fp=100, cost_fn=500, benefit_tp=200):
    """Calculate profit at different thresholds."""
    thresholds = np.arange(0, 1.01, 0.01)
    profits = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        profit = tp * benefit_tp - fp * cost_fp - fn * cost_fn
        profits.append(profit)
    
    best_idx = np.argmax(profits)
    return thresholds, profits, thresholds[best_idx], profits[best_idx]

thresholds, profits, best_t, best_profit = profit_curve(y_test, y_pred_proba_lgbm)
plt.figure(figsize=(10, 6))
plt.plot(thresholds, profits, color='#0f766e', lw=2)
plt.axvline(best_t, color='red', ls='--', label=f'Optimal threshold: {best_t:.2f}')
plt.xlabel('Threshold'); plt.ylabel('Profit (£)')
plt.title(f'Profit Curve — Max Profit: £{best_profit:,.0f} at threshold {best_t:.2f}')
plt.legend()
plt.savefig('../outputs/figures/profit_curve.png', dpi=150, bbox_inches='tight')`,
      },
      {
        title: "2. Model governance checklist",
        detail: "FCA/Basel III uyumlu model governance dokümanı.",
        code: `# docs/governance_checklist.md — içerik olarak oluştur
governance = """# Model Governance Checklist
## FCA SYSC 7.1 & Basel III Compliance

### 1. Model Development
- [x] Business problem clearly defined
- [x] Data quality assessment completed
- [x] Feature engineering documented with business rationale
- [x] Multiple models compared (LR, XGBoost, LightGBM)
- [x] Hyperparameter tuning methodology documented (Optuna)

### 2. Model Validation
- [x] Out-of-sample testing (80/20 stratified split)
- [x] Cross-validation performed (5-fold stratified)
- [x] Discrimination metrics: ROC-AUC, KS, Gini
- [x] Calibration assessment
- [x] Stability testing (PSI on score distributions)

### 3. Explainability & Fairness
- [x] SHAP analysis (global + local explanations)
- [x] Fairness assessment across protected characteristics
- [x] Adverse action reason codes capability
- [x] Regulatory-friendly baseline (Logistic Regression) included

### 4. Ongoing Monitoring
- [ ] Monthly PSI monitoring plan
- [ ] Quarterly model performance review schedule
- [ ] Annual full revalidation plan
- [ ] Champion-challenger framework documented
"""
with open('../docs/governance_checklist.md', 'w') as f:
    f.write(governance)`,
      },
    ],
    commit: "feat: business evaluation with profit curve and regulatory governance checklist",
    checklist: ["Profit curve ile optimal threshold belirlendi", "Business impact parasal olarak hesaplandı", "Model governance checklist oluşturuldu", "docs/governance_checklist.md kaydedildi"]
  },
  {
    day: 10,
    title: "README, Blog Post & Yayınlama",
    date: "Çarşamba",
    hours: "3 saat",
    emoji: "🚀",
    color: "#1d4ed8",
    objectives: [
      "Showcase-ready README yaz",
      "Blog post draft hazırla",
      "Kaggle notebook yayınla",
      "İlk LinkedIn post"
    ],
    tasks: [
      {
        title: "1. README.md — showcase ready",
        detail: "Badges, sonuçlar, görsellerle profesyonel README.",
        code: `# README.md yapısı:
# 🏦 Credit Risk ML Pipeline
# Badges: Python, License, CI status
# 
# ## Overview
# End-to-end credit risk scoring pipeline...
# (business context, regulatory angle)
#
# ## Key Results
# | Model | ROC-AUC | KS | Gini |
# |-------|---------|-----|------|
# | LR    | 0.XXX   | ... | ...  |
# | XGB   | 0.XXX   | ... | ...  |
# | LGBM  | 0.XXX   | ... | ...  |
#
# ## Architecture
# (resim: pipeline diagram)
#
# ## Key Insights
# - DTI ratio is the strongest predictor...
# - SHAP analysis reveals...
#
# ## Regulatory Compliance
# - FCA SYSC aligned governance checklist
# - SHAP-based explainability for GDPR Art. 22
# - Fairness assessment across protected attributes
#
# ## AI Tools Used 🤖
# - GitHub Copilot: boilerplate code generation
# - Claude: code review, debugging, documentation
# - Verification: all AI outputs manually validated
#
# ## Quick Start
# pip install -r requirements.txt
# python -m pytest tests/
#
# ## Project Structure
# (tree diagram)`,
      },
      {
        title: "2. Kaggle notebook & LinkedIn post",
        detail: "EDA notebook'unu Kaggle'a yükle. İlk LinkedIn post yaz.",
        code: `# Kaggle'a yükleme adımları:
# 1. kaggle.com → New Notebook
# 2. 01_eda.ipynb içeriğini temizle ve yükle
# 3. Başlık: "📊 Home Credit - Complete EDA + Feature Engineering"
# 4. Tags: eda, credit-risk, feature-engineering, shap
# 5. Visibility: Public

# LinkedIn post draft:
"""
🚀 Just shipped my first portfolio project: 
Credit Risk ML Pipeline

Built an end-to-end credit scoring system with:
→ Domain-driven feature engineering (DTI, utilization)
→ XGBoost & LightGBM with Optuna tuning
→ SHAP explainability for regulatory compliance
→ Fairness assessment with Fairlearn

Key finding: [en ilginç insight'ını yaz]

Regulatory angle: FCA/Basel III aligned model 
governance checklist included.

GitHub: [link]
Kaggle: [link]
Blog: [link]

#DataScience #MachineLearning #CreditRisk #XAI
"""`,
      },
    ],
    commit: "docs: comprehensive README, model card, and blog post draft",
    checklist: ["README.md profesyonel ve showcase-ready", "AI Tools Used bölümü eklendi", "Kaggle notebook yayınlandı", "LinkedIn post hazır/paylaşıldı", "Blog post draft yazıldı", "Tüm grafikler README'de referans edildi"]
  },
];

export default function SprintPlan() {
  const [activeDay, setActiveDay] = useState(0);
  const [completedTasks, setCompletedTasks] = useState({});
  const [showStructure, setShowStructure] = useState(false);

  const day = DAYS[activeDay];
  const totalTasks = DAYS.reduce((sum, d) => sum + d.checklist.length, 0);
  const completedCount = Object.values(completedTasks).filter(Boolean).length;

  const toggleTask = (key) => {
    setCompletedTasks(prev => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div style={{ fontFamily: "'Newsreader', Georgia, serif", background: "transparent", minHeight: "100vh", color: "#1e293b", padding: "0 0 60px 0" }}>
      
      {/* Header */}
      <div style={{ background: "linear-gradient(135deg, #0f172a 0%, #0f766e 100%)", padding: "36px 24px 28px", borderRadius: "0 0 24px 24px", color: "#f0fdfa", marginBottom: 24 }}>
        <div style={{ fontSize: 11, letterSpacing: 3, textTransform: "uppercase", opacity: 0.5, marginBottom: 6 }}>Proje 1 Sprint Plan</div>
        <h1 style={{ fontSize: 26, fontWeight: 700, margin: 0, lineHeight: 1.2 }}>Credit Risk ML Pipeline</h1>
        <p style={{ fontSize: 13, opacity: 0.7, marginTop: 8 }}>10 günlük detaylı geliştirme planı — kod örnekleri, dosya yapısı ve günlük checklist ile</p>
        
        {/* Progress bar */}
        <div style={{ marginTop: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, opacity: 0.7, marginBottom: 4 }}>
            <span>İlerleme</span>
            <span>{completedCount}/{totalTasks} görev</span>
          </div>
          <div style={{ height: 6, background: "rgba(255,255,255,0.15)", borderRadius: 3 }}>
            <div style={{ height: "100%", width: `${(completedCount/totalTasks)*100}%`, background: "#5eead4", borderRadius: 3, transition: "width 0.3s" }} />
          </div>
        </div>

        <button onClick={() => setShowStructure(!showStructure)} style={{
          marginTop: 14, padding: "6px 16px", borderRadius: 16, fontSize: 12,
          background: "rgba(255,255,255,0.15)", color: "#f0fdfa", border: "1px solid rgba(255,255,255,0.2)",
          cursor: "pointer", fontFamily: "inherit"
        }}>
          {showStructure ? "Yapıyı Gizle" : "Repo Yapısını Göster"}
        </button>
      </div>

      <div style={{ padding: "0 16px", maxWidth: 760, margin: "0 auto" }}>

        {/* Repo structure */}
        {showStructure && (
          <div style={{ background: "#0f172a", color: "#94a3b8", padding: "20px", borderRadius: 14, marginBottom: 20, fontSize: 12, fontFamily: "'Courier New', monospace", whiteSpace: "pre", overflowX: "auto", lineHeight: 1.6 }}>
            {REPO_STRUCTURE}
          </div>
        )}

        {/* Day selector - horizontal scroll */}
        <div style={{ display: "flex", gap: 6, marginBottom: 20, overflowX: "auto", paddingBottom: 8 }}>
          {DAYS.map((d, i) => {
            const dayTasks = d.checklist;
            const dayCompleted = dayTasks.filter((_, ti) => completedTasks[`${i}-${ti}`]).length;
            const isComplete = dayCompleted === dayTasks.length && dayTasks.length > 0;
            return (
              <button key={i} onClick={() => setActiveDay(i)} style={{
                flexShrink: 0, width: 56, padding: "10px 4px", borderRadius: 12, textAlign: "center",
                border: activeDay === i ? `2px solid ${d.color}` : "1px solid #e2e8f0",
                background: activeDay === i ? `${d.color}11` : isComplete ? "#f0fdf4" : "#fff",
                cursor: "pointer", fontFamily: "inherit", transition: "all 0.2s"
              }}>
                <div style={{ fontSize: 18 }}>{d.emoji}</div>
                <div style={{ fontSize: 11, fontWeight: 700, color: d.color, marginTop: 2 }}>G{d.day}</div>
                <div style={{ fontSize: 9, color: "#94a3b8", marginTop: 1 }}>{dayCompleted}/{dayTasks.length}</div>
              </button>
            );
          })}
        </div>

        {/* Active day detail */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
            <span style={{ fontSize: 32 }}>{day.emoji}</span>
            <div>
              <div style={{ fontSize: 20, fontWeight: 700 }}>Gün {day.day}: {day.title}</div>
              <div style={{ fontSize: 13, color: "#64748b" }}>{day.date} — {day.hours}</div>
            </div>
          </div>

          {/* Objectives */}
          <div style={{ background: `${day.color}08`, borderLeft: `3px solid ${day.color}`, padding: "12px 16px", borderRadius: "0 10px 10px 0", marginBottom: 20, marginTop: 12 }}>
            {day.objectives.map((obj, i) => (
              <div key={i} style={{ fontSize: 13, color: "#334155", lineHeight: 1.7 }}>
                <span style={{ color: day.color, marginRight: 6 }}>→</span>{obj}
              </div>
            ))}
          </div>

          {/* Tasks */}
          {day.tasks.map((task, ti) => (
            <div key={ti} style={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: 14, marginBottom: 12, overflow: "hidden" }}>
              <div style={{ padding: "16px 20px" }}>
                <div style={{ fontSize: 15, fontWeight: 700, color: "#0f172a", marginBottom: 4 }}>{task.title}</div>
                <div style={{ fontSize: 13, color: "#64748b", lineHeight: 1.5, marginBottom: 12 }}>{task.detail}</div>
                <div style={{ background: "#0f172a", color: "#e2e8f0", padding: "14px 16px", borderRadius: 10, fontSize: 12, fontFamily: "'Courier New', monospace", whiteSpace: "pre-wrap", overflowX: "auto", lineHeight: 1.5, maxHeight: 300, overflow: "auto" }}>
                  {task.code}
                </div>
              </div>
            </div>
          ))}

          {/* Commit message */}
          <div style={{ background: "#f0fdf4", border: "1px solid #bbf7d0", borderRadius: 10, padding: "10px 16px", marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 14, color: "#16a34a" }}>git</span>
            <code style={{ fontSize: 12, color: "#166534" }}>{day.commit}</code>
          </div>

          {/* Checklist */}
          <div style={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: 14, padding: "16px 20px" }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#94a3b8", letterSpacing: 1, textTransform: "uppercase", marginBottom: 10 }}>Gün Sonu Checklist</div>
            {day.checklist.map((item, ci) => {
              const key = `${activeDay}-${ci}`;
              return (
                <label key={ci} style={{ display: "flex", alignItems: "center", gap: 10, padding: "6px 0", cursor: "pointer", fontSize: 13, color: completedTasks[key] ? "#16a34a" : "#334155" }}>
                  <input type="checkbox" checked={!!completedTasks[key]} onChange={() => toggleTask(key)} style={{ width: 16, height: 16, accentColor: day.color, cursor: "pointer" }} />
                  <span style={{ textDecoration: completedTasks[key] ? "line-through" : "none", opacity: completedTasks[key] ? 0.6 : 1 }}>{item}</span>
                </label>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
