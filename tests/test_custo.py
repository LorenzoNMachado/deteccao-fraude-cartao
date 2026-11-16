import numpy as np
import pytest

from custo import curva_custo, custo_total, melhor_limiar

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
