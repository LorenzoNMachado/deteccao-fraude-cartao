"""Modelos candidatos, espaços de busca e o artefato salvo para uso em produção."""
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from custo import decisao_por_valor
from dados import adicionar_hora


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


def salvar(caminho, pipeline, politica, features, referencia, metadados=None):
    """Salva num único arquivo tudo o que é preciso para decidir sobre uma transação.

    `politica`: {'tipo': 'valor'} (regra por valor, exige probabilidade calibrada)
    ou {'tipo': 'limiar', 'limiar': 0.3}, mais os custos usados.
    `referencia`: amostra das variáveis de treino, base das explicações de modelos lineares.
    """
    joblib.dump({'pipeline': pipeline, 'politica': politica, 'features': list(features),
                 'referencia': referencia[list(features)], 'metadados': metadados or {}}, caminho)


def carregar(caminho):
    """Carrega o artefato salvo por `salvar`.

    joblib usa pickle, que executa código ao carregar: abra só arquivos gerados
    por você mesmo (o padrão do scikit-learn para modelos treinados).
    """
    return joblib.load(caminho)


def decidir(politica, prob, valor):
    """Aplica a política de bloqueio às probabilidades."""
    if politica['tipo'] == 'valor':
        return decisao_por_valor(prob, valor, politica['custo_fp'], politica['custo_fn_fixo'])
    return np.asarray(prob) >= politica['limiar']


def _variaveis(artefato, transacoes):
    if 'Hora' in artefato['features'] and 'Hora' not in transacoes:
        transacoes = adicionar_hora(transacoes)
    return transacoes[artefato['features']]


def prever(artefato, transacoes):
    """Probabilidade de fraude e decisão de bloqueio pela política salva."""
    prob = artefato['pipeline'].predict_proba(_variaveis(artefato, transacoes))[:, 1]
    bloquear = decidir(artefato['politica'], prob, transacoes['Amount'])
    return pd.DataFrame({'prob_fraude': prob, 'bloquear': bloquear}, index=transacoes.index)


def _modelo_base(pipeline):
    # Com CalibratedClassifierCV(ensemble=False) há um único modelo base, treinado em todos os dados
    if hasattr(pipeline, 'calibrated_classifiers_'):
        return pipeline.calibrated_classifiers_[0].estimator
    return pipeline


def contribuicoes(artefato, transacoes):
    """Contribuição de cada variável para o risco de cada transação (valores SHAP).

    Em modelos de árvore usa o TreeExplainer; em modelos lineares, coef × (x − média),
    que é o SHAP exato de um modelo linear. A calibração é monotônica e não muda o sinal
    nem a ordem das contribuições.
    """
    X = _variaveis(artefato, transacoes)
    base = _modelo_base(artefato['pipeline'])
    passos = base.steps if hasattr(base, 'steps') else [('modelo', base)]
    Z, Z_ref = X, artefato['referencia']
    for _, passo in passos[:-1]:
        if hasattr(passo, 'transform'):  # amostradores como o SMOTE só atuam no treino
            Z, Z_ref = passo.transform(Z), passo.transform(Z_ref)
    modelo = passos[-1][1]
    if hasattr(modelo, 'coef_'):
        valores = (np.asarray(Z) - np.asarray(Z_ref).mean(axis=0)) * modelo.coef_[0]
    else:
        import shap
        valores = np.asarray(shap.TreeExplainer(modelo).shap_values(np.asarray(Z, dtype=float)))
        if valores.ndim == 3:  # Random Forest: uma matriz por classe
            valores = valores[..., 1]
    return pd.DataFrame(valores, index=X.index, columns=artefato['features'])


def explicar(artefato, transacoes, n=3):
    """Os `n` fatores que mais aumentam o risco de cada transação, como texto."""
    contrib = contribuicoes(artefato, transacoes)
    X = _variaveis(artefato, transacoes)
    motivos = {}
    for indice, linha in contrib.iterrows():
        maiores = linha[linha > 0].sort_values(ascending=False).head(n)
        textos = [f'{var} = {X.at[indice, var]:,.2f} (+{v:.2f})' for var, v in maiores.items()]
        motivos[indice] = textos + [''] * (n - len(textos))
    return pd.DataFrame.from_dict(motivos, orient='index', columns=[f'Motivo {i + 1}' for i in range(n)])
