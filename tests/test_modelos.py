import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV

from dados import FEATURES, FEATURES_COM_HORA, adicionar_hora
from modelos import candidatos, carregar, contribuicoes, decidir, explicar, prever, salvar

POLITICA_LIMIAR = {'tipo': 'limiar', 'limiar': 0.3, 'custo_fp': 5.0, 'custo_fn_fixo': 20.0}
POLITICA_VALOR = {'tipo': 'valor', 'custo_fp': 5.0, 'custo_fn_fixo': 20.0}


def _artefato(tmp_path, transacoes, nome='Gradient Boosting', features=FEATURES, politica=POLITICA_LIMIAR,
              calibrar=False):
    dados = adicionar_hora(transacoes)
    pipeline = candidatos(n_estimators=10)[nome]
    if calibrar:
        pipeline = CalibratedClassifierCV(pipeline, method='sigmoid', cv=3, ensemble=False)
    pipeline.fit(dados[features], dados['Class'])
    caminho = tmp_path / 'modelo.joblib'
    salvar(caminho, pipeline, politica, features, dados.head(100))
    return carregar(caminho)


def test_smote_so_no_treino(transacoes):
    pipeline = candidatos(n_estimators=10)['Random Forest + SMOTE']
    assert 'smote' in pipeline.named_steps
    pipeline.fit(transacoes[FEATURES], transacoes['Class'])
    # Na previsão o SMOTE não atua: uma probabilidade por transação recebida
    assert pipeline.predict_proba(transacoes[FEATURES].head(50)).shape == (50, 2)


def test_artefato_aplica_o_limiar_salvo(tmp_path, transacoes):
    artefato = _artefato(tmp_path, transacoes)
    resultado = prever(artefato, transacoes)
    assert artefato['politica'] == POLITICA_LIMIAR
    assert (resultado['bloquear'] == (resultado['prob_fraude'] >= 0.3)).all()


def test_artefato_aplica_regra_por_valor(tmp_path, transacoes):
    artefato = _artefato(tmp_path, transacoes, politica=POLITICA_VALOR)
    resultado = prever(artefato, transacoes)
    esperado = resultado['prob_fraude'] * (transacoes['Amount'] + 20) > (1 - resultado['prob_fraude']) * 5
    assert (resultado['bloquear'] == esperado).all()


def test_decidir_limiar():
    assert decidir(POLITICA_LIMIAR, np.array([0.2, 0.3]), np.array([10, 10])).tolist() == [False, True]


def test_prever_calcula_a_hora_quando_o_modelo_usa(tmp_path, transacoes):
    artefato = _artefato(tmp_path, transacoes, features=FEATURES_COM_HORA)
    assert 'Hora' not in transacoes
    assert len(prever(artefato, transacoes)) == len(transacoes)


def test_prever_ignora_colunas_extras_e_ordem(tmp_path, transacoes):
    artefato = _artefato(tmp_path, transacoes)
    embaralhada = transacoes[['Class', 'Time'] + FEATURES[::-1]]
    np.testing.assert_allclose(prever(artefato, embaralhada)['prob_fraude'],
                               prever(artefato, transacoes)['prob_fraude'])


@pytest.mark.parametrize('nome', ['Gradient Boosting', 'Random Forest + SMOTE', 'Regressão Logística'])
def test_contribuicoes_de_cada_tipo_de_modelo(tmp_path, transacoes, nome):
    artefato = _artefato(tmp_path, transacoes, nome=nome, calibrar=True)
    fraudes = transacoes[transacoes['Class'] == 1].head(5)
    contrib = contribuicoes(artefato, fraudes)
    assert contrib.shape == (5, len(FEATURES))
    # As fraudes sintéticas têm V14 deslocado para baixo: V14 deve empurrar o risco para cima
    assert (contrib['V14'] > 0).mean() >= 0.8


def test_shap_soma_a_saida_do_modelo(tmp_path, transacoes):
    artefato = _artefato(tmp_path, transacoes)
    modelo = artefato['pipeline'][-1]
    X = transacoes[FEATURES].head(20)
    contrib = contribuicoes(artefato, X)
    import shap
    base = shap.TreeExplainer(modelo).expected_value
    np.testing.assert_allclose(contrib.sum(axis=1) + base, modelo.decision_function(X), atol=1e-4)


def test_explicar_lista_motivos(tmp_path, transacoes):
    artefato = _artefato(tmp_path, transacoes, calibrar=True)
    fraudes = transacoes[transacoes['Class'] == 1].head(3)
    motivos = explicar(artefato, fraudes, n=2)
    assert list(motivos.columns) == ['Motivo 1', 'Motivo 2']
    assert motivos['Motivo 1'].str.contains(r'\(\+').all()
