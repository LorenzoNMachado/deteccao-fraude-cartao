import pandas as pd
import pytest

from dados import carregar, remover_duplicadas, separar_temporal, validar


def test_separacao_temporal_sem_sobreposicao(transacoes):
    treino, validacao, teste = separar_temporal(transacoes.sample(frac=1, random_state=1))
    assert treino['Time'].max() <= validacao['Time'].min()
    assert validacao['Time'].max() <= teste['Time'].min()
    assert len(treino) + len(validacao) + len(teste) == len(transacoes)
    assert len(teste) == pytest.approx(0.2 * len(transacoes), abs=1)


def test_remove_duplicadas(transacoes):
    com_copias = pd.concat([transacoes, transacoes.head(10)])
    assert len(remover_duplicadas(com_copias)) == len(transacoes)


def test_validar_aponta_coluna_ausente(transacoes):
    with pytest.raises(ValueError, match='Amount'):
        validar(transacoes.drop(columns='Amount'))


def test_validar_rejeita_classe_invalida(transacoes):
    transacoes.loc[0, 'Class'] = 2
    with pytest.raises(ValueError, match='0 e 1'):
        validar(transacoes)


def test_carregar_do_banco(tmp_path, transacoes):
    # SQLite passa pelo mesmo caminho do SQLAlchemy usado com o PostgreSQL
    url = f'sqlite:///{tmp_path / "fraude.db"}'
    transacoes.to_sql('transactions', url, index=False)
    lido = carregar(database_url=url)
    pd.testing.assert_frame_equal(lido, transacoes, check_dtype=False)


def test_carregar_do_csv(tmp_path, transacoes):
    caminho = tmp_path / 'creditcard.csv'
    transacoes.to_csv(caminho, index=False)
    assert carregar(csv=caminho).shape == transacoes.shape


def test_carregar_sem_fonte():
    with pytest.raises(ValueError):
        carregar()
