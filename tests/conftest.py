import numpy as np
import pandas as pd
import pytest

from dados import FEATURES


@pytest.fixture
def transacoes():
    """Transações sintéticas: fraudes deslocadas em V14 e V10, como no dataset real."""
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=FEATURES[:-1])
    df['Amount'] = rng.exponential(80, n).round(2)
    df['Class'] = (rng.random(n) < 0.03).astype(int)
    df.loc[df['Class'] == 1, ['V14', 'V10']] -= 3
    df.insert(0, 'Time', np.sort(rng.uniform(0, 172800, n)).round())
    return df
