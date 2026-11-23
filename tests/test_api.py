import json

import pytest
from fastapi.testclient import TestClient

from api import MAX_REQUEST_BYTES
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


def test_prever_aceita_lote_com_exatamente_mil_transacoes(cliente, transacoes):
    legitima = transacoes[transacoes['Class'] == 0].head(1).copy()
    legitima['Amount'] = 0.0
    transacao = _corpo(legitima)[0]

    resposta = cliente.post('/prever', json=[transacao] * 1000)

    assert resposta.status_code == 200
    assert len(resposta.json()) == 1000


def test_prever_rejeita_transacao_incompleta(cliente, transacoes):
    corpo = _corpo(transacoes.head(1))
    del corpo[0]['V14']
    resposta = cliente.post('/prever', json=corpo)
    assert resposta.status_code == 422
    assert 'input' not in resposta.json()['detail'][0]


def test_prever_rejeita_lista_vazia(cliente):
    assert cliente.post('/prever', json=[]).status_code == 422


@pytest.mark.parametrize(
    ('campo', 'valor', 'marcador'),
    [
        ('Amount', -1, '-1'),
        ('Amount', float('nan'), 'NaN'),
        ('Amount', float('inf'), 'Infinity'),
        ('Amount', float('-inf'), '-Infinity'),
        ('Time', -1, '-1'),
        ('V1', float('inf'), 'Infinity'),
    ],
)
def test_prever_rejeita_valores_invalidos(cliente, transacoes, campo, valor, marcador):
    corpo = _corpo(transacoes.head(1))
    corpo[0][campo] = valor
    resposta = cliente.post(
        '/prever',
        content=json.dumps(corpo, allow_nan=True),
        headers={'content-type': 'application/json'},
    )
    assert resposta.status_code == 422
    detalhe = resposta.json()['detail'][0]
    assert set(detalhe) == {'loc', 'msg', 'type'}
    assert 'input' not in detalhe
    assert marcador not in resposta.text


@pytest.mark.parametrize(('campo', 'valor'), [('Amount', 1e100), ('V1', 1e100)])
def test_prever_aceita_valores_finitos_muito_elevados(cliente, transacoes, campo, valor):
    corpo = _corpo(transacoes[transacoes['Class'] == 0].head(1))
    corpo[0]['Amount'] = 0.0
    corpo[0][campo] = valor

    resposta = cliente.post('/prever', json=corpo)

    assert resposta.status_code == 200
    assert len(resposta.json()) == 1
    assert 0 <= resposta.json()[0]['prob_fraude'] <= 1


def test_prever_rejeita_corpo_maior_que_limite_de_bytes(cliente):
    resposta = cliente.post(
        '/prever',
        content=b' ' * (MAX_REQUEST_BYTES + 1),
        headers={'content-type': 'application/json'},
    )

    assert resposta.status_code == 413
    assert resposta.json() == {'detail': 'Corpo da requisição excede o limite de 2 MiB.'}


def test_prever_rejeita_lote_maior_que_limite(cliente, transacoes):
    transacao = _corpo(transacoes.head(1))[0]
    resposta = cliente.post('/prever', json=[transacao] * 1001)
    assert resposta.status_code == 422
