"""Carrega o creditcard.csv na tabela `transactions` do PostgreSQL.

A conexão vem de DATABASE_URL (no ambiente ou no .env; veja .env.example).

    python load_data.py                    # usa data/creditcard.csv
    python load_data.py caminho/do.csv
"""
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from dados import TABELA, validar  # noqa: E402


def main():
    load_dotenv()
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        sys.exit('Defina DATABASE_URL (copie .env.example para .env e ajuste a senha).')
    csv = sys.argv[1] if len(sys.argv) > 1 else 'data/creditcard.csv'

    df = pd.read_csv(csv)
    validar(df)
    engine = create_engine(database_url)
    try:
        df.to_sql(TABELA, engine, if_exists='replace', index=False, chunksize=10_000)
    finally:
        engine.dispose()
    print(f'{len(df):,} transações gravadas na tabela "{TABELA}".')


if __name__ == '__main__':
    main()
