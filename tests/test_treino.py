import json

from treino import treinar


def test_fluxo_completo(tmp_path, transacoes):
    metricas = treinar(transacoes, tmp_path / 'reports', tmp_path / 'models', n_iter=2, cv_splits=2,
                       nomes_modelos=['Regressão Logística', 'Gradient Boosting'], n_estimators=10)

    assert metricas['modelo'] in ('Regressão Logística', 'Gradient Boosting')
    assert 0 < metricas['limiar'] < 1
    assert (tmp_path / 'models' / 'modelo_fraude.joblib').exists()
    for figura in ['pr_curve', 'custo_vs_limiar', 'prob_distribution', 'confusion_matrix', 'feature_importance']:
        assert (tmp_path / 'reports' / 'figures' / f'{figura}.png').exists()
    salvas = json.loads((tmp_path / 'reports' / 'metricas.json').read_text(encoding='utf-8'))
    assert salvas['teste']['limiar_custo']['limiar'] == metricas['limiar']
