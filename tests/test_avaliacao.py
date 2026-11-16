import numpy as np
import pytest

from avaliacao import bootstrap_ic, metricas_decisao

y = np.array([1, 1, 0, 0, 0, 0])
bloquear = np.array([True, False, True, False, False, False])
valor = np.array([100.0, 50.0, 30.0, 10.0, 5.0, 1.0])


def test_metricas_decisao():
    m = metricas_decisao(y, bloquear, valor)
    assert m['recall'] == 0.5
    assert m['precision'] == 0.5
    assert (m['fraudes_perdidas'], m['bloqueios_indevidos']) == (1, 1)
    # Sem modelo: 100 + 50 + 2 × 20; com modelo: 50 + 20 + 5
    assert m['custo_sem_modelo'] == pytest.approx(190)
    assert m['custo'] == pytest.approx(75)
    assert m['economia'] == pytest.approx(1 - 75 / 190)


def test_bootstrap_contem_a_estimativa_pontual():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    prob = np.clip(y * 0.6 + rng.random(2000) * 0.5, 0, 1)
    bloquear = prob > 0.5
    valor = rng.exponential(50, 2000)
    pontual = metricas_decisao(y, bloquear, valor, prob)
    ic = bootstrap_ic(y, bloquear, valor, prob, n=200)
    for nome in ['recall', 'precision', 'economia', 'pr_auc']:
        baixo, alto = ic[nome]
        assert baixo <= pontual[nome] <= alto
        assert alto - baixo < 0.2


def test_bootstrap_reprodutivel():
    assert bootstrap_ic(y, bloquear, valor, n=50, seed=1) == bootstrap_ic(y, bloquear, valor, n=50, seed=1)
