# Imagem da API. Treine antes (python train_model.py) para gerar models/modelo_fraude.joblib.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api.py .
COPY models/modelo_fraude.joblib models/

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
