import json

import pandas as pd

from modelos import carregar, prever
from treino import treinar


def test_fluxo_completo(tmp_path, transacoes):
    metricas = treinar(transacoes, tmp_path / 'reports', tmp_path / 'models', n_iter=2, cv_splits=2,
                       nomes_modelos=['Regressão Logística', 'Gradient Boosting'], n_estimators=10, n_bootstrap=50)

    assert metricas['modelo'] in ('Regressão Logística', 'Gradient Boosting')
    assert metricas['politica']['tipo'] in ('valor', 'limiar')
    for nome in ['recall', 'precision', 'economia', 'pr_auc']:
        baixo, alto = metricas['teste_ic95'][nome]
        assert baixo <= metricas['teste']['politica_escolhida'][nome] <= alto
    for figura in ['pr_curve', 'custo_vs_limiar', 'decisao_no_teste', 'confusion_matrix', 'feature_importance']:
        assert (tmp_path / 'reports' / 'figures' / f'{figura}.png').exists()
    salvas = json.loads((tmp_path / 'reports' / 'metricas.json').read_text(encoding='utf-8'))
    assert salvas['politica'] == metricas['politica']
    assert len(pd.read_csv(tmp_path / 'reports' / 'alertas_exemplo.csv')) > 0

    # O modelo de produção usa todas as transações e decide com a política escolhida
    artefato = carregar(tmp_path / 'models' / 'modelo_fraude.joblib')
    assert artefato['metadados']['transacoes_treino'] == len(transacoes.drop_duplicates())
    assert artefato['politica'] == metricas['politica']
    assert prever(artefato, transacoes.head(10))['bloquear'].dtype == bool
