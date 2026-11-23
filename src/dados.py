"""Carga, limpeza e separação temporal das transações."""
import pandas as pd
from sqlalchemy import create_engine

FEATURES = [f'V{i}' for i in range(1, 29)] + ['Amount']
FEATURES_COM_HORA = FEATURES + ['Hora']
ALVO = 'Class'
TABELA = 'transactions'


def carregar(database_url=None, csv=None):
    """Lê as transações do banco (se `database_url` for informado) ou do CSV."""
    if database_url:
        engine = create_engine(database_url)
        try:
            return pd.read_sql_table(TABELA, engine)
        finally:
            engine.dispose()
    if csv:
        return pd.read_csv(csv)
    raise ValueError('Informe DATABASE_URL ou o caminho do CSV.')


def validar(df):
    """Confere colunas, tipos e valores; levanta erro com a causa se algo estiver errado."""
    faltando = set(FEATURES + ['Time', ALVO]) - set(df.columns)
    if faltando:
        raise ValueError(f'Colunas ausentes: {sorted(faltando)}')
    if df[FEATURES + ['Time', ALVO]].isna().any().any():
        raise ValueError('Há valores nulos nas transações.')
    if not set(df[ALVO].unique()) <= {0, 1}:
        raise ValueError(f'{ALVO} deve conter apenas 0 e 1.')


def adicionar_hora(df):
    """Hora do dia (0 a 24) a partir de `Time`, em segundos desde a primeira transação."""
    novo = df.copy()
    novo['Hora'] = (novo['Time'] % 86_400) / 3_600
    return novo


def remover_duplicadas(df):
    """Remove transações repetidas, que poderiam cair ao mesmo tempo no treino e no teste."""
    return df.drop_duplicates().reset_index(drop=True)


def separar_temporal(df, frac_validacao=0.2, frac_teste=0.2):
    """Treino com as transações mais antigas, validação no meio e teste com as mais recentes."""
    ordenado = df.sort_values('Time', kind='stable').reset_index(drop=True)
    n = len(ordenado)
    fim_treino = int(n * (1 - frac_validacao - frac_teste))
    fim_validacao = int(n * (1 - frac_teste))
    return (ordenado.iloc[:fim_treino],
            ordenado.iloc[fim_treino:fim_validacao],
            ordenado.iloc[fim_validacao:])
