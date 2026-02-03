from sqlalchemy import create_engine
import pandas as pd

engine = create_engine(
    "postgresql+psycopg2://usuario:senha@localhost:5432/fraud_db"
)

csvfile = "csvfile/creditcard.csv"
df = pd.read_csv(csvfile)

df.to_sql("transactions",
          engine,
          if_exists="replace",
          index=False)

print("✅ Banco de dados atualizado com todas as colunas!")