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

# 2 MiB accommodates 1,000 records with all 30 numeric fields represented by
# the longest finite IEEE-754 values, plus JSON framing and ordinary whitespace.
MAX_REQUEST_BYTES = 2 * 1024 * 1024
ERRO_CORPO_GRANDE = 'Corpo da requisição excede o limite de 2 MiB.'


class _CorpoMuitoGrande(Exception):
    pass


class LimiteDeCorpo:
    """Enforce a byte cap for /prever, including chunked request bodies."""

    def __init__(self, app, limite_bytes):
        self.app = app
        self.limite_bytes = limite_bytes

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] != '/prever' or scope['method'] != 'POST':
            await self.app(scope, receive, send)
            return

        content_length = next(
            (value for key, value in scope.get('headers', []) if key.lower() == b'content-length'),
            None,
        )
        if content_length is not None:
            try:
                if int(content_length) > self.limite_bytes:
                    await JSONResponse(status_code=413, content={'detail': ERRO_CORPO_GRANDE})(
                        scope, receive, send
                    )
                    return
            except ValueError:
                # Let the ASGI server or JSON parser report a malformed header/body.
                pass

        total_recebido = 0
        resposta_iniciada = False

        async def receber_limitado():
            nonlocal total_recebido
            mensagem = await receive()
            if mensagem['type'] == 'http.request':
                total_recebido += len(mensagem.get('body', b''))
                if total_recebido > self.limite_bytes:
                    raise _CorpoMuitoGrande
            return mensagem

        async def enviar_rastreado(mensagem):
            nonlocal resposta_iniciada
            if mensagem['type'] == 'http.response.start':
                resposta_iniciada = True
            await send(mensagem)

        try:
            await self.app(scope, receber_limitado, enviar_rastreado)
        except _CorpoMuitoGrande:
            if resposta_iniciada:
                raise
            await JSONResponse(status_code=413, content={'detail': ERRO_CORPO_GRANDE})(
                scope, receive, send
            )

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
app.add_middleware(LimiteDeCorpo, limite_bytes=MAX_REQUEST_BYTES)


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


@app.post(
    '/prever',
    response_model=list[Resultado],
    responses={413: {'description': 'O corpo da requisição excede 2 MiB.'}},
)
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
