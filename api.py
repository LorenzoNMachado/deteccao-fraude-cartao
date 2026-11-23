"""API de detecção de fraude.

    uvicorn api:app --reload               # http://localhost:8000/docs

O modelo vem de MODELO_PATH (padrão: models/modelo_fraude.joblib), gerado por train_model.py.
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, create_model

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from modelos import carregar, explicar, prever  # noqa: E402

Transacao = create_model(
    'Transacao',
    Time=(float, Field(ge=0, allow_inf_nan=False,
                       description='Segundos desde a primeira transação do período (usado para a hora do dia)')),
    Amount=(float, Field(ge=0, allow_inf_nan=False, description='Valor da transação em euros')),
    **{f'V{i}': (float, Field(..., allow_inf_nan=False)) for i in range(1, 29)},
)


class Resultado(BaseModel):
    prob_fraude: float
    bloquear: bool
    motivos: list[str] = Field(description='Fatores que mais aumentaram o risco (só para transações bloqueadas)')


estado = {}


@asynccontextmanager
async def ciclo_de_vida(app):
    caminho = os.getenv('MODELO_PATH', 'models/modelo_fraude.joblib')
    if not Path(caminho).exists():
        raise RuntimeError(f'Modelo não encontrado em {caminho}. Rode train_model.py antes.')
    estado['artefato'] = carregar(caminho)
    yield
    estado.clear()


app = FastAPI(title='Detecção de Fraude em Cartões', lifespan=ciclo_de_vida)


@app.exception_handler(RequestValidationError)
async def tratar_erro_de_validacao(_request: Request, exc: RequestValidationError):
    # Do not echo invalid values; NaN/Infinity would otherwise make JSONResponse fail.
    detalhes = [
        {'loc': erro['loc'], 'msg': erro['msg'], 'type': erro['type']}
        for erro in exc.errors()
    ]
    return JSONResponse(status_code=422, content={'detail': detalhes})


@app.get('/saude')
def saude():
    artefato = estado['artefato']
    return {'status': 'ok', 'politica': artefato['politica'], 'modelo': artefato['metadados']}


@app.post('/prever', response_model=list[Resultado])
def prever_transacoes(transacoes: Annotated[list[Transacao], Body(max_length=1000)]):
    if not transacoes:
        raise HTTPException(status_code=422, detail='Envie ao menos uma transação.')
    artefato = estado['artefato']
    df = pd.DataFrame([t.model_dump() for t in transacoes])
    resultado = prever(artefato, df)
    motivos = pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)
    bloqueadas = resultado.index[resultado['bloquear']]
    if len(bloqueadas):
        textos = explicar(artefato, df.loc[bloqueadas])
        for i, linha in textos.iterrows():
            motivos[i] = [m for m in linha if m]
    return [Resultado(prob_fraude=float(r.prob_fraude), bloquear=bool(r.bloquear), motivos=motivos[i])
            for i, r in resultado.iterrows()]
