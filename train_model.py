"""Treina o modelo de detecção de fraude.

Lê as transações do PostgreSQL quando DATABASE_URL está definida (no ambiente
ou no .env); caso contrário, lê o CSV.

    python train_model.py                      # banco (DATABASE_URL) ou data/creditcard.csv
    python train_model.py --csv outro.csv      # força a leitura de um CSV
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from dados import carregar  # noqa: E402
from treino import treinar  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', help='caminho do CSV (ignora DATABASE_URL)')
    parser.add_argument('--n-iter', type=int, default=10, help='combinações testadas na busca de hiperparâmetros')
    parser.add_argument('--bootstrap', type=int, default=1000, help='reamostragens para os intervalos de confiança')
    parser.add_argument('--relatorios', default='reports', help='pasta de métricas e gráficos')
    parser.add_argument('--modelos', default='models', help='pasta do modelo salvo')
    args = parser.parse_args()

    load_dotenv()
    database_url = None if args.csv else os.getenv('DATABASE_URL')
    csv = args.csv or 'data/creditcard.csv'
    try:
        df = carregar(database_url=database_url, csv=csv)
    except Exception as erro:
        origem = 'banco (DATABASE_URL)' if database_url else csv
        sys.exit(f'Erro ao ler as transações de {origem}: {erro}')
    print(f'Dados carregados de {"banco" if database_url else csv}: {df.shape}')

    treinar(df, args.relatorios, args.modelos, n_iter=args.n_iter, n_bootstrap=args.bootstrap)


if __name__ == '__main__':
    main()
