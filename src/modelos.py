"""Modelos candidatos, espaços de busca e o artefato salvo para uso em produção."""
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from dados import FEATURES


def candidatos(random_state=42, n_estimators=300):
    """Pipelines completos (pré-processamento + modelo).

    O SMOTE fica dentro do pipeline do imblearn: na validação cruzada ele é
    aplicado só na parte de treino de cada fold, e nunca na hora de prever.
    """
    return {
        'Regressão Logística': Pipeline([
            ('escala', RobustScaler()),
            ('modelo', LogisticRegression(class_weight='balanced', max_iter=2000)),
        ]),
        'Random Forest': Pipeline([
            ('modelo', RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced_subsample',
                                              min_samples_leaf=2, random_state=random_state, n_jobs=-1)),
        ]),
        'Random Forest + SMOTE': Pipeline([
            ('escala', RobustScaler()),
            # Fraudes sintéticas até 10% das legítimas: balancear 50/50 dobraria o treino
            ('smote', SMOTE(sampling_strategy=0.1, random_state=random_state)),
            ('modelo', RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=2,
                                              random_state=random_state, n_jobs=-1)),
        ]),
        # Sem regularização, o boosting fica instável com 0,2% de fraudes (PR-AUC ~0,2)
        'Gradient Boosting': Pipeline([
            ('modelo', HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, early_stopping=False,
                                                      l2_regularization=1.0, min_samples_leaf=50,
                                                      random_state=random_state)),
        ]),
    }


ESPACOS_BUSCA = {
    'Regressão Logística': {'modelo__C': loguniform(1e-3, 1e2)},
    'Random Forest': {
        'modelo__max_depth': [10, 20, None],
        'modelo__min_samples_leaf': randint(1, 6),
        'modelo__max_features': ['sqrt', 'log2', 0.5],
    },
    'Random Forest + SMOTE': {
        'smote__sampling_strategy': [0.05, 0.1, 0.3],
        'modelo__max_depth': [10, 20, None],
        'modelo__min_samples_leaf': randint(1, 6),
        'modelo__max_features': ['sqrt', 'log2', 0.5],
    },
    'Gradient Boosting': {
        'modelo__learning_rate': loguniform(0.02, 0.2),
        'modelo__max_leaf_nodes': randint(15, 48),
        'modelo__min_samples_leaf': randint(50, 400),
        'modelo__l2_regularization': uniform(0.5, 4.5),
    },
}


def salvar(caminho, pipeline, limiar, metadados=None):
    """Salva pipeline, limiar e lista de variáveis num único arquivo."""
    joblib.dump({'pipeline': pipeline, 'limiar': limiar, 'features': FEATURES,
                 'metadados': metadados or {}}, caminho)


def carregar(caminho):
    """Carrega o artefato salvo por `salvar`.

    joblib usa pickle, que executa código ao carregar: abra só arquivos gerados
    por você mesmo (o padrão do scikit-learn para modelos treinados).
    """
    return joblib.load(caminho)


def prever(artefato, transacoes):
    """Probabilidade de fraude e decisão de bloqueio com o limiar salvo."""
    prob = artefato['pipeline'].predict_proba(transacoes[artefato['features']])[:, 1]
    return pd.DataFrame({'prob_fraude': prob, 'bloquear': prob >= artefato['limiar']},
                        index=transacoes.index)
