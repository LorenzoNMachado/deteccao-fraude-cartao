import pytest

from sintetico import gerar_transacoes


@pytest.fixture
def transacoes():
    return gerar_transacoes()
