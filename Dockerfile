# Imagem da API. Treine antes (python train_model.py) para gerar models/modelo_fraude.joblib.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup --system app && adduser --system --ingroup app app
COPY --chown=app:app src/ src/
COPY --chown=app:app api.py .
COPY --chown=app:app models/modelo_fraude.joblib models/

USER app
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
