import numpy as np

from dados import FEATURES
from modelos import candidatos, carregar, prever, salvar


def test_smote_so_no_treino(transacoes):
    pipeline = candidatos(n_estimators=10)['Random Forest + SMOTE']
    assert 'smote' in pipeline.named_steps
    pipeline.fit(transacoes[FEATURES], transacoes['Class'])
    # Na previsão o SMOTE não atua: uma probabilidade por transação recebida
    assert pipeline.predict_proba(transacoes[FEATURES].head(50)).shape == (50, 2)


def test_artefato_guarda_pipeline_e_limiar(tmp_path, transacoes):
    pipeline = candidatos(n_estimators=10)['Gradient Boosting'].fit(transacoes[FEATURES], transacoes['Class'])
    caminho = tmp_path / 'modelo.joblib'
    salvar(caminho, pipeline, limiar=0.3)

    artefato = carregar(caminho)
    resultado = prever(artefato, transacoes)
    assert artefato['limiar'] == 0.3
    assert (resultado['bloquear'] == (resultado['prob_fraude'] >= 0.3)).all()


def test_prever_ignora_colunas_extras_e_ordem(tmp_path, transacoes):
    pipeline = candidatos(n_estimators=10)['Gradient Boosting'].fit(transacoes[FEATURES], transacoes['Class'])
    salvar(tmp_path / 'm.joblib', pipeline, limiar=0.5)
    artefato = carregar(tmp_path / 'm.joblib')
    embaralhada = transacoes[['Class', 'Time'] + FEATURES[::-1]]
    np.testing.assert_allclose(prever(artefato, embaralhada)['prob_fraude'],
                               prever(artefato, transacoes)['prob_fraude'])
