"""Transações sintéticas no formato do creditcard.csv, para testes e para o CI.

    python src/sintetico.py data/creditcard.csv
"""
import sys

import numpy as np
import pandas as pd


def gerar_transacoes(n=3000, taxa_fraude=0.03, seed=0):
    """Fraudes deslocadas em V14 e V10, como no dataset real; Time em ordem crescente."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f'V{i}' for i in range(1, 29)])
    df['Amount'] = rng.exponential(80, n).round(2)
    df['Class'] = (rng.random(n) < taxa_fraude).astype(int)
    df.loc[df['Class'] == 1, ['V14', 'V10']] -= 3
    df.insert(0, 'Time', np.sort(rng.uniform(0, 172_800, n)).round())
    return df


if __name__ == '__main__':
    destino = sys.argv[1] if len(sys.argv) > 1 else 'data/creditcard.csv'
    gerar_transacoes().to_csv(destino, index=False)
    print(f'Transações sintéticas gravadas em {destino}')
