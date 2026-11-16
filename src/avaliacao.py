"""Métricas de uma política de bloqueio e intervalos de confiança por bootstrap."""
import numpy as np
from sklearn.metrics import average_precision_score

from custo import custo_decisao


def metricas_decisao(y, bloquear, valor, prob=None):
    """Recall, precision, erros, custo e economia em relação a não bloquear nada."""
    y, bloquear, valor = np.asarray(y), np.asarray(bloquear, dtype=bool), np.asarray(valor, dtype=float)
    tp = int(((y == 1) & bloquear).sum())
    fn = int(((y == 1) & ~bloquear).sum())
    fp = int(((y == 0) & bloquear).sum())
    custo = custo_decisao(y, bloquear, valor)
    sem_modelo = custo_decisao(y, np.zeros_like(bloquear), valor)
    metricas = {
        'recall': tp / (tp + fn) if tp + fn else 0.0,
        'precision': tp / (tp + fp) if tp + fp else 0.0,
        'fraudes_detectadas': tp,
        'fraudes_perdidas': fn,
        'bloqueios_indevidos': fp,
        'custo': custo,
        'custo_sem_modelo': sem_modelo,
        'economia': 1 - custo / sem_modelo if sem_modelo else 0.0,
    }
    if prob is not None:
        metricas['pr_auc'] = float(average_precision_score(y, prob))
    return metricas


def bootstrap_ic(y, bloquear, valor, prob=None, n=1000, confianca=0.95, seed=42):
    """Intervalo percentil de cada métrica, reamostrando as transações com reposição."""
    y, bloquear, valor = np.asarray(y), np.asarray(bloquear, dtype=bool), np.asarray(valor, dtype=float)
    prob = None if prob is None else np.asarray(prob)
    rng = np.random.default_rng(seed)
    amostras = {}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        m = metricas_decisao(y[i], bloquear[i], valor[i], None if prob is None else prob[i])
        for nome, v in m.items():
            amostras.setdefault(nome, []).append(v)
    alfa = (1 - confianca) / 2
    return {nome: (float(np.quantile(v, alfa)), float(np.quantile(v, 1 - alfa))) for nome, v in amostras.items()}
