# =========================
# IMPORTS
# =========================
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler

from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# =========================
# BANCO + DADOS
# =========================
try:
    engine = create_engine(
        "postgresql+psycopg2://usuario:senha@localhost:5432/fraud_db"
    )
    query = '''SELECT * FROM transactions'''
    df = pd.read_sql(query, engine)
    print(f"Dados carregados: {df.shape}")
except Exception as e:
    print(f"Erro ao conectar ao banco: {e}")
    exit(1)

print(f"Colunas: {df.columns.tolist()}")
print(f"\nDistribuição de classes:")
print(df['Class'].value_counts())
print(f"Proporção de fraude: {df['Class'].mean()*100:.2f}%")

# =========================
# FEATURES / TARGET
# =========================
v_features = [f'V{i}' for i in range(1, 29)]
features_selecionadas = v_features + ['Amount']
X = df[features_selecionadas]
y = df['Class']

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print(f"\nTrain: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# =========================
# Validação dos dados
# =========================
print(f"\n=== VALIDAÇÃO DOS DADOS ===")
print(f"Valores nulos: {df.isnull().sum().sum()}")
print(f"Valores duplicados: {df.duplicated().sum()}")
print(f"Valores infinitos: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")

assert df["Class"].dtype in [np.int64, np.int32], "Class deve ser int"
assert all(df.drop('Class', axis=1).dtypes != 'object'), "Features devem ser numéricas"

# =========================
# PRÉ-PROCESSAMENTO
# =========================
print("\n=== PRÉ-PROCESSAMENTO ===")
scaler = RobustScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val_scaled   = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

print(f"Treino após SMOTE: {np.bincount(y_train_res)}")

# =========================
# BASELINE DUMMY
# =========================
print("\n" + "="*50)
print("BASELINE DUMMY")
print("="*50)

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train_scaled, y_train)

dummy_proba = dummy.predict_proba(X_test_scaled)[:, 1]
dummy_pred = dummy.predict(X_test_scaled)

print("Recall   :", recall_score(y_test, dummy_pred))
print("Precision:", precision_score(y_test, dummy_pred, zero_division=0))
print("F1       :", f1_score(y_test, dummy_pred))
print("ROC-AUC  :", roc_auc_score(y_test, dummy_proba))
print("PR-AUC   :", average_precision_score(y_test, dummy_proba))

# =========================
# LOGISTIC REGRESSION
# =========================
print("\n" + "="*50)
print("LOGISTIC REGRESSION")
print("="*50)

log_reg = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

log_reg.fit(X_train_scaled, y_train)

lr_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
lr_pred = log_reg.predict(X_test_scaled)

print(classification_report(y_test, lr_pred, target_names=['Legítimo', 'Fraude']))
print("ROC-AUC:", roc_auc_score(y_test, lr_proba))
print("PR-AUC :", average_precision_score(y_test, lr_proba))

# =========================
# RANDOM FOREST
# =========================
print("\n" + "="*50)
print("RANDOM FOREST - MODELO SMOTE")
print("="*50)

rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_res, y_train_res)

rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
rf_pred = rf.predict(X_test_scaled)

print("=== Threshold 0.5 ===")
print("Recall   :", recall_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("F1       :", f1_score(y_test, rf_pred))
print("ROC-AUC  :", roc_auc_score(y_test, rf_proba))
print("PR-AUC   :", average_precision_score(y_test, rf_proba))

# =========================
#  VALIDAÇÃO CRUZADA
# =========================
print("\n=== Validação Cruzada (5-fold) ===")
cv_recall = cross_val_score(rf, X_train, y_train, cv=5, scoring='recall')
cv_precision = cross_val_score(rf, X_train, y_train, cv=5, scoring='precision')
cv_f1 = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')

print(f"Recall    : {cv_recall.mean():.3f} (+/- {cv_recall.std():.3f})")
print(f"Precision : {cv_precision.mean():.3f} (+/- {cv_precision.std():.3f})")
print(f"F1-Score  : {cv_f1.mean():.3f} (+/- {cv_f1.std():.3f})")

# =========================
# THRESHOLD TUNING
# =========================
print("\n" + "="*50)
print("THRESHOLD TUNING")
print("="*50)

thresholds = np.arange(0.01, 0.9, 0.01)

best_f1 = 0
best_threshold = 0.5

rf_proba_val = rf.predict_proba(X_val_scaled)[:, 1]
rf_proba_test = rf.predict_proba(X_test_scaled)[:, 1] 

for t in thresholds:
    preds = (rf_proba_val >= t).astype(int)  
    f1 = f1_score(y_val, preds)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Threshold otimizado: {best_threshold:.2f}")
print(f"F1-Score máximo (validation): {best_f1:.4f}")

final_pred = (rf_proba_test >= best_threshold).astype(int)

print("\n=== Random Forest (Threshold Otimizado) ===")
print("Recall   :", recall_score(y_test, final_pred))
print("Precision:", precision_score(y_test, final_pred))
print("F1       :", f1_score(y_test, final_pred))

# =========================
# ANÁLISE DE CUSTO
# =========================
print("\n" + "="*50)
print("ANÁLISE DE CUSTO DE NEGÓCIO")
print("="*50)

CUSTO_FN = 2500
CUSTO_FP = 10

custos = []
for t in thresholds:
    preds = (rf_proba >= t).astype(int)
    
    fn = ((y_test == 1) & (preds == 0)).sum()
    fp = ((y_test == 0) & (preds == 1)).sum()
    
    custo = (fn * CUSTO_FN) + (fp * CUSTO_FP)
    custos.append((t, custo, fn, fp))

best_custo = min(custos, key=lambda x: x[1])
print(f"\nThreshold ótimo por custo: {best_custo[0]}")
print(f"Custo total: R$ {best_custo[1]:,.2f}")
print(f"Fraudes perdidas (FN): {best_custo[2]}")
print(f"Clientes bloqueados (FP): {best_custo[3]}")

pred_custo = (rf_proba >= best_custo[0]).astype(int)
print("\n=== Métricas com Threshold de Custo ===")
print("Recall   :", recall_score(y_test, pred_custo))
print("Precision:", precision_score(y_test, pred_custo))
print("F1       :", f1_score(y_test, pred_custo))

# =========================
# ANÁLISE DE ERROS
# =========================
print("\n" + "="*50)
print("ANÁLISE DE ERROS")
print("="*50)

fn_mask = (y_test == 1) & (final_pred == 0)
fp_mask = (y_test == 0) & (final_pred == 1)

print(f"\nFalsos Negativos (fraudes perdidas): {fn_mask.sum()}")
print(f"Falsos Positivos (clientes bloqueados): {fp_mask.sum()}")

# =========================
# TUNAGEM DE HIPERPARÂMETROS
# =========================
print("\n" + "="*50)
print("TUNAGEM DE HIPERPARÂMETROS")
print("="*50)

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [20, 30, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=5,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Buscando melhores hiperparâmetros... (pode demorar)")
# TREINAR SEM ESCALAR
rf_random.fit(X_train_res, y_train_res)

print(f"\nMelhores parâmetros: {rf_random.best_params_}")
print(f"Melhor recall (CV): {rf_random.best_score_:.4f}")

rf_tuned = rf_random.best_estimator_
rf_tuned_proba = rf_tuned.predict_proba(X_test_scaled)[:, 1]
rf_tuned_pred = (rf_tuned_proba >= best_threshold).astype(int)

print("\n=== Random Forest TUNADO ===")
print("Recall   :", recall_score(y_test, rf_tuned_pred))
print("Precision:", precision_score(y_test, rf_tuned_pred))
print("F1       :", f1_score(y_test, rf_tuned_pred))

# =========================
# VISUALIZAÇÕES
# =========================
print("\n" + "="*50)
print("GERANDO VISUALIZAÇÕES")
print("="*50)

precision, recall, _ = precision_recall_curve(y_test, rf_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, linewidth=2)
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.title("Curva Precision-Recall", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pr_curve.png', dpi=300)
plt.show()

cm = confusion_matrix(y_test, final_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legítimo', 'Fraude'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Matriz de Confusão')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(rf_proba[y_test == 0], bins=50, alpha=0.6, label='Legítimas', color='blue')
plt.hist(rf_proba[y_test == 1], bins=50, alpha=0.6, label='Fraudes', color='red')
plt.axvline(best_threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold={best_threshold}')
plt.xlabel('Probabilidade de Fraude', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.legend(fontsize=10)
plt.title('Distribuição de Probabilidades Preditas', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('prob_distribution.png', dpi=300)
plt.show()

#Métricas vs Threshold
recalls = []
precisions = []
f1s = []

for t in np.arange(0.01, 1.0, 0.01):
    preds = (rf_proba >= t).astype(int)
    recalls.append(recall_score(y_test, preds))
    precisions.append(precision_score(y_test, preds, zero_division=0))
    f1s.append(f1_score(y_test, preds, zero_division=0))

plt.figure(figsize=(10, 6))
plt.plot(np.arange(0.01, 1.0, 0.01), recalls, label='Recall', linewidth=2)
plt.plot(np.arange(0.01, 1.0, 0.01), precisions, label='Precision', linewidth=2)
plt.plot(np.arange(0.01, 1.0, 0.01), f1s, label='F1-Score', linewidth=2)
plt.axvline(best_threshold, color='red', linestyle='--', alpha=0.7, label='Threshold Ótimo')
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.legend(fontsize=10)
plt.title('Métricas vs Threshold', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('metrics_vs_threshold.png', dpi=300)
plt.show()

#feature Importance
importances = pd.Series(
    rf_tuned.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
importances[:20].plot(kind='barh')
plt.xlabel('Importância', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 20 Features Mais Importantes', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()

print("\nTop 10 Features:")
print(importances.head(10))

# =========================
# SALVAR MODELO
# =========================
print("\n" + "="*50)
print("SALVANDO MODELO")
print("="*50)

joblib.dump(rf_tuned, 'modelo_fraude_rf_tuned.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump({'threshold': best_threshold}, 'config.pkl')

print("Modelo salvo em 'modelo_fraude_rf_tuned.pkl'")
print("Scaler salvo em 'scaler.pkl'")
print("Configurações salvas em 'config.pkl'")

print("\n" + "="*50)
print("TREINAMENTO CONCLUÍDO!")
print("="*50)