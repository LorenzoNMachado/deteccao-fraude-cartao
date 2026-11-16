import pytest
from fastapi.testclient import TestClient

from dados import FEATURES, adicionar_hora
from modelos import candidatos, salvar

POLITICA = {'tipo': 'valor', 'custo_fp': 5.0, 'custo_fn_fixo': 20.0}


@pytest.fixture
def cliente(tmp_path, transacoes, monkeypatch):
    dados = adicionar_hora(transacoes)
    pipeline = candidatos(n_estimators=10)['Gradient Boosting'].fit(dados[FEATURES], dados['Class'])
    salvar(tmp_path / 'modelo.joblib', pipeline, POLITICA, FEATURES, dados.head(50), {'modelo': 'teste'})
    monkeypatch.setenv('MODELO_PATH', str(tmp_path / 'modelo.joblib'))
    import api
    with TestClient(api.app) as c:
        yield c


def _corpo(transacoes):
    return transacoes.drop(columns='Class').to_dict(orient='records')


def test_saude(cliente):
    resposta = cliente.get('/saude')
    assert resposta.status_code == 200
    assert resposta.json()['politica'] == POLITICA


def test_prever_bloqueia_fraude_e_explica(cliente, transacoes):
    fraude = transacoes[transacoes['Class'] == 1].nlargest(1, 'Amount')
    legitima = transacoes[transacoes['Class'] == 0].head(1)
    resposta = cliente.post('/prever', json=_corpo(fraude) + _corpo(legitima))
    assert resposta.status_code == 200
    r_fraude, r_legitima = resposta.json()
    assert r_fraude['bloquear'] and r_fraude['motivos']
    assert not r_legitima['bloquear'] and r_legitima['motivos'] == []


def test_prever_rejeita_transacao_incompleta(cliente, transacoes):
    corpo = _corpo(transacoes.head(1))
    del corpo[0]['V14']
    assert cliente.post('/prever', json=corpo).status_code == 422


def test_prever_rejeita_lista_vazia(cliente):
    assert cliente.post('/prever', json=[]).status_code == 422
