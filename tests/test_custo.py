import numpy as np
import pytest

from custo import curva_custo, custo_decisao, custo_total, decisao_por_valor, melhor_limiar

y = np.array([1, 1, 0, 0, 0])
prob = np.array([0.9, 0.2, 0.6, 0.1, 0.05])
valor = np.array([100.0, 50.0, 30.0, 10.0, 5.0])


def test_custo_usa_valor_da_fraude_perdida():
    # Limiar 0,5: perde a fraude de €50 (50 + 20 fixo) e bloqueia 1 legítima (5)
    assert custo_total(y, prob, valor, 0.5, custo_fp=5, custo_fn_fixo=20) == pytest.approx(75)


def test_custo_sem_bloqueios_e_o_valor_de_todas_as_fraudes():
    assert custo_total(y, prob, valor, 1.01, custo_fp=5, custo_fn_fixo=20) == pytest.approx(150 + 40)


def test_curva_conta_erros():
    curva = curva_custo(y, prob, valor, [0.15, 0.5], custo_fp=5, custo_fn_fixo=20)
    assert curva['fraudes_perdidas'].tolist() == [0, 1]
    assert curva['bloqueios_indevidos'].tolist() == [1, 1]


def test_melhor_limiar_minimiza_custo():
    # 0,15 pega as duas fraudes com 1 bloqueio indevido (custo 5) e é o mínimo
    assert melhor_limiar(y, prob, valor, [0.15, 0.5, 0.95], custo_fp=5, custo_fn_fixo=20) == 0.15


def test_fp_caro_desloca_o_limiar_para_cima():
    assert melhor_limiar(y, prob, valor, [0.15, 0.5, 0.95], custo_fp=1000, custo_fn_fixo=0) == 0.95


def test_regra_por_valor_bloqueia_transacao_cara_com_risco_baixo():
    # Liberar €2.000 com 1% de risco: perda esperada 0,01 × 2.020 = 20,2 > 0,99 × 5
    assert decisao_por_valor([0.01], [2000], custo_fp=5, custo_fn_fixo=20)[0]


def test_regra_por_valor_libera_transacao_barata_com_mesmo_risco():
    # Liberar €3 com 1% de risco: 0,01 × 23 = 0,23 < 0,99 × 5
    assert not decisao_por_valor([0.01], [3], custo_fp=5, custo_fn_fixo=20)[0]


def test_regra_por_valor_tem_fronteira_no_ponto_de_indiferenca():
    # p × (v + F) = (1 − p) × C  ->  p = C / (v + F + C)
    p = 5 / (80 + 20 + 5)
    assert not decisao_por_valor([p - 1e-6], [80])[0]
    assert decisao_por_valor([p + 1e-6], [80])[0]


def test_custo_decisao_igual_ao_custo_por_limiar():
    assert custo_decisao(y, prob >= 0.5, valor) == custo_total(y, prob, valor, 0.5)
