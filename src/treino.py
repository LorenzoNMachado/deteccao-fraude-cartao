"""Fluxo completo: seleção de modelo, ajuste, limiar por custo, avaliação e gráficos.

Regra geral: o conjunto de teste (transações mais recentes) é usado uma única
vez, no final. Modelo e hiperparâmetros são escolhidos por validação cruzada
temporal no treino; o limiar, no conjunto de validação.
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # só salva arquivos; nunca abre janela
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score

from custo import CUSTO_FN_FIXO, CUSTO_FP, curva_custo, custo_total, melhor_limiar
from dados import ALVO, FEATURES, remover_duplicadas, separar_temporal, validar
from modelos import ESPACOS_BUSCA, candidatos, salvar

LIMIARES = np.round(np.arange(0.01, 1.0, 0.01), 2)
AZUL, LARANJA, CINZA, TEXTO_SECUNDARIO = '#2a78d6', '#eb6834', '#8a8a85', '#55554f'
plt.rcParams.update({
    'figure.dpi': 110,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': '#e4e4e0',
    'grid.linewidth': 0.8,
})


def _log(msg):
    print(msg, flush=True)


def _metricas_no_limiar(y, prob, valor, limiar):
    pred = prob >= limiar
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'limiar': float(limiar),
        'recall': recall_score(y, pred, zero_division=0),
        'precision': precision_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'fraudes_detectadas': int(tp),
        'fraudes_perdidas': int(fn),
        'bloqueios_indevidos': int(fp),
        'custo': custo_total(y, prob, valor, limiar),
    }


def treinar(df, pasta_relatorios, pasta_modelos, n_iter=10, cv_splits=3, nomes_modelos=None,
            n_estimators=300, random_state=42):
    inicio = time.time()
    pasta_relatorios, pasta_modelos = Path(pasta_relatorios), Path(pasta_modelos)
    pasta_figuras = pasta_relatorios / 'figures'
    pasta_figuras.mkdir(parents=True, exist_ok=True)
    pasta_modelos.mkdir(parents=True, exist_ok=True)

    # --- Dados ---
    validar(df)
    duplicadas = int(df.duplicated().sum())
    df = remover_duplicadas(df)
    treino, validacao, teste = separar_temporal(df)
    X_tr, y_tr = treino[FEATURES], treino[ALVO]
    X_va, y_va = validacao[FEATURES], validacao[ALVO]
    X_te, y_te = teste[FEATURES], teste[ALVO]
    _log(f'{len(df):,} transações ({duplicadas:,} duplicadas removidas) | fraudes: {int(df[ALVO].sum())}')
    for nome, parte in [('Treino', treino), ('Validação', validacao), ('Teste', teste)]:
        _log(f'  {nome:<10} {len(parte):>7,} transações | {int(parte[ALVO].sum()):>3} fraudes | '
             f'Time {parte["Time"].min():,.0f} a {parte["Time"].max():,.0f} s')

    # --- 1. Comparação de modelos (PR-AUC na validação cruzada temporal do treino) ---
    # A validação tem poucas fraudes; a CV no treino usa mais fraudes e dá uma escolha mais estável
    _log(f'\n1. Comparação de modelos (PR-AUC, {cv_splits} folds temporais no treino)')
    cv = TimeSeriesSplit(n_splits=cv_splits)
    modelos = candidatos(random_state, n_estimators)
    if nomes_modelos:
        modelos = {n: modelos[n] for n in nomes_modelos}
    comparacao = {}
    for nome, modelo in modelos.items():
        t0 = time.time()
        notas = cross_val_score(modelo, X_tr, y_tr, cv=cv, scoring='average_precision')
        comparacao[nome] = {'pr_auc_cv': notas.mean(), 'pr_auc_cv_desvio': notas.std(),
                            'segundos': round(time.time() - t0, 1)}
        _log(f'  {nome:<24} PR-AUC {notas.mean():.4f} (± {notas.std():.4f}) | {comparacao[nome]["segundos"]:.0f}s')
    melhor = max(comparacao, key=lambda n: comparacao[n]['pr_auc_cv'])
    _log(f'  Melhor: {melhor}')

    # --- 2. Ajuste de hiperparâmetros (mesmos folds; SMOTE, se houver, dentro do pipeline) ---
    _log(f'\n2. Ajuste de hiperparâmetros de {melhor} ({n_iter} combinações)')
    busca = RandomizedSearchCV(clone(modelos[melhor]), ESPACOS_BUSCA[melhor], n_iter=n_iter, cv=cv,
                               scoring='average_precision', random_state=random_state, n_jobs=1)
    busca.fit(X_tr, y_tr)
    _log(f'  Melhores parâmetros: {busca.best_params_}')
    _log(f'  PR-AUC na CV: {busca.best_score_:.4f} (sem ajuste: {comparacao[melhor]["pr_auc_cv"]:.4f})')
    usar_ajustado = busca.best_score_ > comparacao[melhor]['pr_auc_cv']
    final = busca.best_estimator_ if usar_ajustado else clone(modelos[melhor]).fit(X_tr, y_tr)
    pr_auc_validacao = average_precision_score(y_va, final.predict_proba(X_va)[:, 1])
    _log(f'  Modelo final: {melhor} {"ajustado" if usar_ajustado else "com parâmetros padrão"} '
         f'| PR-AUC na validação: {pr_auc_validacao:.4f}')

    # --- 3. Limiar por custo, escolhido na validação ---
    _log(f'\n3. Limiar por custo na validação (FN = valor da transação + €{CUSTO_FN_FIXO:.0f}; FP = €{CUSTO_FP:.0f})')
    prob_va = final.predict_proba(X_va)[:, 1]
    curva_va = curva_custo(y_va, prob_va, X_va['Amount'], LIMIARES)
    limiar = melhor_limiar(y_va, prob_va, X_va['Amount'], LIMIARES)
    _log(f'  Limiar escolhido: {limiar:.2f}')

    # --- 4. Avaliação final no teste (uma única vez) ---
    _log('\n4. Avaliação no teste')
    prob_te = final.predict_proba(X_te)[:, 1]
    valor_te = X_te['Amount'].to_numpy()
    sem_modelo = custo_total(y_te, np.zeros(len(y_te)), valor_te, 0.5)
    teste_metricas = {
        'pr_auc': average_precision_score(y_te, prob_te),
        'roc_auc': roc_auc_score(y_te, prob_te),
        'fraudes': int(y_te.sum()),
        'transacoes': int(len(y_te)),
        'valor_fraudes': float(valor_te[y_te.to_numpy() == 1].sum()),
        'custo_sem_modelo': sem_modelo,
        'limiar_custo': _metricas_no_limiar(y_te, prob_te, valor_te, limiar),
        'limiar_0_5': _metricas_no_limiar(y_te, prob_te, valor_te, 0.5),
    }
    for nome, m in [(f'limiar {limiar:.2f}', teste_metricas['limiar_custo']), ('limiar 0,50', teste_metricas['limiar_0_5'])]:
        _log(f'  {nome}: recall {m["recall"]:.3f} | precision {m["precision"]:.3f} | '
             f'perdidas {m["fraudes_perdidas"]} | bloqueios indevidos {m["bloqueios_indevidos"]} | custo €{m["custo"]:,.2f}')
    _log(f'  Sem modelo (nenhum bloqueio): custo €{sem_modelo:,.2f} | PR-AUC {teste_metricas["pr_auc"]:.4f}')

    # --- 5. Importância por permutação (validação) ---
    _log('\n5. Importância por permutação (queda de PR-AUC na validação)')
    perm = permutation_importance(final, X_va, y_va, scoring='average_precision', n_repeats=5,
                                  random_state=random_state, n_jobs=1)
    importancia = (np.round(perm.importances_mean, 5), np.round(perm.importances_std, 5))
    ordem = np.argsort(importancia[0])[::-1]
    _log('  ' + ', '.join(f'{FEATURES[i]} ({importancia[0][i]:.3f})' for i in ordem[:5]))

    # --- Gráficos ---
    _grafico_pr(y_te, prob_te, limiar, teste_metricas, pasta_figuras / 'pr_curve.png')
    _grafico_custo(curva_va, limiar, pasta_figuras / 'custo_vs_limiar.png')
    _grafico_probabilidades(y_te, prob_te, limiar, pasta_figuras / 'prob_distribution.png')
    _grafico_matriz(y_te, prob_te >= limiar, limiar, pasta_figuras / 'confusion_matrix.png')
    _grafico_importancia(importancia, ordem[:15], pasta_figuras / 'feature_importance.png')

    # --- Artefatos ---
    metricas = {
        'dados': {'transacoes': len(df), 'duplicadas_removidas': duplicadas,
                  'treino': len(treino), 'validacao': len(validacao), 'teste': len(teste)},
        'custos': {'fp': CUSTO_FP, 'fn_fixo': CUSTO_FN_FIXO, 'fn_variavel': 'Amount'},
        'comparacao_validacao': comparacao,
        'modelo': melhor,
        'ajustado': bool(usar_ajustado),
        'busca': {'melhores_parametros': {k: (v if isinstance(v, (int, float, str, type(None))) else float(v))
                                          for k, v in busca.best_params_.items()},
                  'pr_auc_cv': busca.best_score_},
        'pr_auc_validacao': pr_auc_validacao,
        'limiar': limiar,
        'teste': teste_metricas,
        'importancia_permutacao': {FEATURES[i]: float(importancia[0][i]) for i in ordem},
        'segundos': round(time.time() - inicio),
    }
    salvar(pasta_modelos / 'modelo_fraude.joblib', final, limiar,
           {'modelo': melhor, 'ajustado': bool(usar_ajustado), 'custos': metricas['custos']})
    (pasta_relatorios / 'metricas.json').write_text(json.dumps(metricas, indent=2, ensure_ascii=False, default=float),
                                                   encoding='utf-8')
    _log(f'\nConcluído em {metricas["segundos"]}s. Modelo em {pasta_modelos / "modelo_fraude.joblib"}')
    return metricas


def _salvar(fig, caminho):
    fig.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _grafico_pr(y, prob, limiar, m, caminho):
    precisao, recall, _ = precision_recall_curve(y, prob)
    ponto = m['limiar_custo']
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precisao, color=AZUL, linewidth=2)
    ax.axhline(y.mean(), color=CINZA, linestyle='--', linewidth=1)
    ax.annotate(f'sem modelo ({y.mean():.2%} de fraudes)', xy=(0, y.mean()), xycoords=('axes fraction', 'data'),
                xytext=(4, 4), textcoords='offset points', ha='left', fontsize=8, color=TEXTO_SECUNDARIO)
    ax.plot(ponto['recall'], ponto['precision'], 'o', color=AZUL, markersize=8, markeredgecolor='white',
            markeredgewidth=2)
    ax.annotate(f'limiar {limiar:.2f}', (ponto['recall'], ponto['precision']), xytext=(-10, -16),
                textcoords='offset points', ha='right', fontsize=9)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel('Recall (fraudes detectadas)')
    ax.set_ylabel('Precision (alertas que eram fraude)')
    ax.set_title(f'Curva Precision-Recall no teste (PR-AUC {m["pr_auc"]:.3f})', loc='left')
    _salvar(fig, caminho)


def _grafico_custo(curva, limiar, caminho):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(curva['limiar'], curva['custo'], color=AZUL, linewidth=2)
    minimo = curva.loc[curva['limiar'] == limiar, 'custo'].iloc[0]
    ax.plot(limiar, minimo, 'o', color=AZUL, markersize=8, markeredgecolor='white', markeredgewidth=2)
    ax.annotate(f'mínimo em {limiar:.2f} (€{minimo:,.0f})', (limiar, minimo), xytext=(0, 12),
                textcoords='offset points', ha='center', fontsize=9)
    ax.set_xlabel('Limiar de decisão')
    ax.set_ylabel('Custo total (€)')
    ax.set_title('Custo por limiar na validação', loc='left')
    _salvar(fig, caminho)


def _grafico_probabilidades(y, prob, limiar, caminho):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 51)
    ax.hist(prob[y == 0], bins=bins, color=AZUL, alpha=0.75, label='Legítimas', edgecolor='white', linewidth=0.5)
    ax.hist(prob[y == 1], bins=bins, color=LARANJA, alpha=0.85, label='Fraudes', edgecolor='white', linewidth=0.5)
    ax.axvline(limiar, color=CINZA, linestyle='--', linewidth=1)
    ax.annotate(f'limiar {limiar:.2f}', xy=(limiar, 1), xycoords=('data', 'axes fraction'), xytext=(4, -12),
                textcoords='offset points', fontsize=8, color=TEXTO_SECUNDARIO)
    ax.set_yscale('log')
    ax.set_xlabel('Probabilidade de fraude prevista')
    ax.set_ylabel('Transações (escala log)')
    ax.set_title('Distribuição das probabilidades no teste', loc='left')
    ax.legend(frameon=False)
    _salvar(fig, caminho)


def _grafico_matriz(y, pred, limiar, caminho):
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ConfusionMatrixDisplay(confusion_matrix(y, pred), display_labels=['Legítima', 'Fraude']).plot(
        ax=ax, cmap='Blues', values_format=',d', colorbar=False)
    ax.grid(False)
    ax.set_xlabel('Previsto')
    ax.set_ylabel('Real')
    ax.set_title(f'Matriz de confusão no teste (limiar {limiar:.2f})', loc='left')
    _salvar(fig, caminho)


def _grafico_importancia(importancia, indices, caminho):
    media, desvio = importancia
    nomes = [FEATURES[i] for i in indices][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(nomes, media[indices][::-1], xerr=desvio[indices][::-1], color=AZUL, height=0.6,
            error_kw={'ecolor': TEXTO_SECUNDARIO, 'capsize': 2, 'elinewidth': 1})
    ax.set_xlabel('Queda da PR-AUC ao embaralhar a variável')
    ax.set_title('Importância por permutação (validação)', loc='left')
    ax.grid(axis='y', visible=False)
    _salvar(fig, caminho)
