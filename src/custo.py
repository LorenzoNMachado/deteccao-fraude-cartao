"""Custo de negócio das decisões do modelo, usando o valor real de cada transação."""
import numpy as np
import pandas as pd

# Valores hipotéticos, em euros (moeda do dataset); ajuste para a realidade do negócio
CUSTO_FP = 5.0         # contato com o cliente e atrito de um bloqueio indevido
CUSTO_FN_FIXO = 20.0   # custo operacional de uma contestação, além do valor perdido


def custo_total(y, prob, valor, limiar, custo_fp=CUSTO_FP, custo_fn_fixo=CUSTO_FN_FIXO):
    """Custo de bloquear as transações com prob >= limiar.

    Fraude não bloqueada (FN) custa o valor da transação mais o custo fixo;
    transação legítima bloqueada (FP) custa `custo_fp`.
    """
    y, prob, valor = np.asarray(y), np.asarray(prob), np.asarray(valor, dtype=float)
    bloqueada = prob >= limiar
    fn = (y == 1) & ~bloqueada
    fp = (y == 0) & bloqueada
    return float(valor[fn].sum() + fn.sum() * custo_fn_fixo + fp.sum() * custo_fp)


def curva_custo(y, prob, valor, limiares, **custos):
    """Custo, fraudes perdidas e bloqueios indevidos para cada limiar."""
    y, prob = np.asarray(y), np.asarray(prob)
    linhas = []
    for limiar in limiares:
        bloqueada = prob >= limiar
        linhas.append({
            'limiar': limiar,
            'custo': custo_total(y, prob, valor, limiar, **custos),
            'fraudes_perdidas': int(((y == 1) & ~bloqueada).sum()),
            'bloqueios_indevidos': int(((y == 0) & bloqueada).sum()),
        })
    return pd.DataFrame(linhas)


def melhor_limiar(y, prob, valor, limiares, **custos):
    """Limiar de menor custo (em caso de empate, o menor limiar)."""
    curva = curva_custo(y, prob, valor, limiares, **custos)
    return float(curva.loc[curva['custo'].idxmin(), 'limiar'])
