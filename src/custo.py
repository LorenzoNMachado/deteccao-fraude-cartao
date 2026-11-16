"""Custo de negócio das decisões do modelo, usando o valor real de cada transação."""
import numpy as np
import pandas as pd

# Valores hipotéticos, em euros (moeda do dataset); ajuste para a realidade do negócio
CUSTO_FP = 5.0         # contato com o cliente e atrito de um bloqueio indevido
CUSTO_FN_FIXO = 20.0   # custo operacional de uma contestação, além do valor perdido


def custo_decisao(y, bloquear, valor, custo_fp=CUSTO_FP, custo_fn_fixo=CUSTO_FN_FIXO):
    """Custo de um conjunto de decisões de bloqueio.

    Fraude não bloqueada (FN) custa o valor da transação mais o custo fixo;
    transação legítima bloqueada (FP) custa `custo_fp`.
    """
    y, bloquear, valor = np.asarray(y), np.asarray(bloquear, dtype=bool), np.asarray(valor, dtype=float)
    fn = (y == 1) & ~bloquear
    fp = (y == 0) & bloquear
    return float(valor[fn].sum() + fn.sum() * custo_fn_fixo + fp.sum() * custo_fp)


def custo_total(y, prob, valor, limiar, custo_fp=CUSTO_FP, custo_fn_fixo=CUSTO_FN_FIXO):
    """Custo de bloquear as transações com prob >= limiar."""
    return custo_decisao(y, np.asarray(prob) >= limiar, valor, custo_fp, custo_fn_fixo)


def decisao_por_valor(prob, valor, custo_fp=CUSTO_FP, custo_fn_fixo=CUSTO_FN_FIXO):
    """Bloqueia quando a perda esperada de liberar supera o custo de bloquear.

    Liberar custa prob × (valor + custo fixo) em média; bloquear custa (1 − prob) × custo_fp.
    Exige probabilidades calibradas.
    """
    prob, valor = np.asarray(prob), np.asarray(valor, dtype=float)
    return prob * (valor + custo_fn_fixo) > (1 - prob) * custo_fp


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
