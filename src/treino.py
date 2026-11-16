"""Fluxo completo: seleção de modelo, ajuste, calibração, política de bloqueio, avaliação e gráficos.

Regra geral: o conjunto de teste (transações mais recentes) é usado uma única
vez para avaliar. Modelo, hiperparâmetros e variáveis são escolhidos por
validação cruzada temporal no treino; a política de bloqueio, no conjunto de
validação. Depois da avaliação, o modelo de produção é retreinado com todos os
dados, mantendo todas essas escolhas.
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # só salva arquivos; nunca abre janela
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, brier_score_loss, confusion_matrix,
                             precision_recall_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit, cross_val_score

from avaliacao import bootstrap_ic, metricas_decisao
from custo import CUSTO_FN_FIXO, CUSTO_FP, curva_custo, custo_decisao, decisao_por_valor, melhor_limiar
from dados import (ALVO, FEATURES, FEATURES_COM_HORA, adicionar_hora, remover_duplicadas,
                   separar_temporal, validar)
from modelos import ESPACOS_BUSCA, candidatos, decidir, explicar, salvar

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


def _nome_politica(politica):
    return 'regra por valor' if politica['tipo'] == 'valor' else f'limiar {politica["limiar"]:.2f}'


def _resumo(nome, m):
    return (f'  {nome:<18} recall {m["recall"]:.3f} | precision {m["precision"]:.3f} | '
            f'perdidas {m["fraudes_perdidas"]} | bloqueios indevidos {m["bloqueios_indevidos"]} | '
            f'custo €{m["custo"]:,.2f} (economia {m["economia"]:.1%})')


def treinar(df, pasta_relatorios, pasta_modelos, n_iter=10, cv_splits=3, nomes_modelos=None,
            n_estimators=300, n_bootstrap=1000, random_state=42):
    inicio = time.time()
    pasta_relatorios, pasta_modelos = Path(pasta_relatorios), Path(pasta_modelos)
    pasta_figuras = pasta_relatorios / 'figures'
    pasta_figuras.mkdir(parents=True, exist_ok=True)
    pasta_modelos.mkdir(parents=True, exist_ok=True)

    # --- Dados ---
    validar(df)
    duplicadas = int(df.duplicated().sum())
    df = adicionar_hora(remover_duplicadas(df))
    treino, validacao, teste = separar_temporal(df)
    y_tr, y_va, y_te = treino[ALVO], validacao[ALVO], teste[ALVO]
    valor_va, valor_te = validacao['Amount'].to_numpy(), teste['Amount'].to_numpy()
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
        notas = cross_val_score(modelo, treino[FEATURES], y_tr, cv=cv, scoring='average_precision')
        comparacao[nome] = {'pr_auc_cv': notas.mean(), 'pr_auc_cv_desvio': notas.std(),
                            'segundos': round(time.time() - t0, 1)}
        _log(f'  {nome:<24} PR-AUC {notas.mean():.4f} (± {notas.std():.4f}) | {comparacao[nome]["segundos"]:.0f}s')
    melhor = max(comparacao, key=lambda n: comparacao[n]['pr_auc_cv'])
    _log(f'  Melhor: {melhor}')

    # --- 2. Ajuste de hiperparâmetros (mesmos folds; SMOTE, se houver, dentro do pipeline) ---
    _log(f'\n2. Ajuste de hiperparâmetros de {melhor} ({n_iter} combinações)')
    busca = RandomizedSearchCV(clone(modelos[melhor]), ESPACOS_BUSCA[melhor], n_iter=n_iter, cv=cv,
                               scoring='average_precision', random_state=random_state, n_jobs=1, refit=False)
    busca.fit(treino[FEATURES], y_tr)
    _log(f'  Melhores parâmetros: {busca.best_params_}')
    _log(f'  PR-AUC na CV: {busca.best_score_:.4f} (sem ajuste: {comparacao[melhor]["pr_auc_cv"]:.4f})')
    usar_ajustado = busca.best_score_ > comparacao[melhor]['pr_auc_cv']
    config = clone(modelos[melhor])
    if usar_ajustado:
        config.set_params(**busca.best_params_)
    pr_auc_cv = max(busca.best_score_, comparacao[melhor]['pr_auc_cv'])
    _log(f'  Configuração: {melhor} {"ajustado" if usar_ajustado else "com parâmetros padrão"}')

    # --- 3. Hora do dia como variável ---
    _log('\n3. Hora do dia como variável (mesmos folds)')
    notas_hora = cross_val_score(clone(config), treino[FEATURES_COM_HORA], y_tr, cv=cv, scoring='average_precision')
    usar_hora = notas_hora.mean() > pr_auc_cv
    features = FEATURES_COM_HORA if usar_hora else FEATURES
    _log(f'  PR-AUC com hora: {notas_hora.mean():.4f} | sem hora: {pr_auc_cv:.4f} -> '
         f'{"usa" if usar_hora else "não usa"} a hora')

    # --- 4. Calibração das probabilidades (CV no treino; mantida só se melhorar na validação) ---
    # A regra por valor usa a probabilidade como chance real de fraude; a calibração
    # sigmoide corrige modelos que super ou subestimam o risco
    _log('\n4. Calibração (sigmoide)')

    def modelo_calibrado():
        # ensemble=False: um único modelo treinado em todos os dados + calibrador ajustado fora da amostra
        return CalibratedClassifierCV(clone(config), method='sigmoid', ensemble=False,
                                      cv=StratifiedKFold(5, shuffle=True, random_state=random_state))

    X_tr, X_va, X_te = treino[features], validacao[features], teste[features]
    sem_calibracao = clone(config).fit(X_tr, y_tr)
    calibrado = modelo_calibrado().fit(X_tr, y_tr)
    p_va_sem = sem_calibracao.predict_proba(X_va)[:, 1]
    p_va_cal = calibrado.predict_proba(X_va)[:, 1]
    usar_calibracao = brier_score_loss(y_va, p_va_cal) < brier_score_loss(y_va, p_va_sem)
    modelo, p_va = (calibrado, p_va_cal) if usar_calibracao else (sem_calibracao, p_va_sem)
    novo_modelo = modelo_calibrado if usar_calibracao else (lambda: clone(config))
    calibracao = {
        'fraudes_reais_validacao': int(y_va.sum()),
        'fraudes_previstas_sem_calibracao': float(p_va_sem.sum()),
        'fraudes_previstas_calibrado': float(p_va_cal.sum()),
        'brier_sem_calibracao': brier_score_loss(y_va, p_va_sem),
        'brier_calibrado': brier_score_loss(y_va, p_va_cal),
        'usa_calibracao': bool(usar_calibracao),
    }
    _log(f'  Fraudes na validação: {calibracao["fraudes_reais_validacao"]} reais | previstas (soma das '
         f'probabilidades): {calibracao["fraudes_previstas_sem_calibracao"]:.1f} sem calibração, '
         f'{calibracao["fraudes_previstas_calibrado"]:.1f} calibrado')
    _log(f'  Brier: {calibracao["brier_sem_calibracao"]:.6f} sem calibração, {calibracao["brier_calibrado"]:.6f} '
         f'calibrado -> {"usa" if usar_calibracao else "não usa"} a calibração')

    # --- 5. Política de bloqueio, escolhida na validação ---
    _log(f'\n5. Política de bloqueio na validação (FN = valor + €{CUSTO_FN_FIXO:.0f}; FP = €{CUSTO_FP:.0f})')
    curva_va = curva_custo(y_va, p_va, valor_va, LIMIARES)
    limiar = melhor_limiar(y_va, p_va, valor_va, LIMIARES)
    custo_limiar_va = float(curva_va.loc[curva_va['limiar'] == limiar, 'custo'].iloc[0])
    custo_valor_va = custo_decisao(y_va, decisao_por_valor(p_va, valor_va), valor_va)
    custos = {'custo_fp': CUSTO_FP, 'custo_fn_fixo': CUSTO_FN_FIXO}
    politica = ({'tipo': 'valor', **custos} if custo_valor_va <= custo_limiar_va
                else {'tipo': 'limiar', 'limiar': limiar, **custos})
    _log(f'  Melhor limiar único ({limiar:.2f}): €{custo_limiar_va:,.2f} | regra por valor: €{custo_valor_va:,.2f}')
    _log(f'  Política escolhida: {_nome_politica(politica)}')

    # --- 6. Avaliação no teste (uma única vez) ---
    _log(f'\n6. Avaliação no teste (IC 95% por bootstrap, {n_bootstrap} reamostragens)')
    p_te = modelo.predict_proba(X_te)[:, 1]
    bloquear_te = decidir(politica, p_te, valor_te)
    decisoes = {
        'politica_escolhida': bloquear_te,
        'regra_por_valor': decisao_por_valor(p_te, valor_te),
        f'limiar_{limiar:.2f}': p_te >= limiar,
        'limiar_0.50': p_te >= 0.5,
    }
    avaliacao = {nome: metricas_decisao(y_te, b, valor_te, p_te) for nome, b in decisoes.items()}
    ic = bootstrap_ic(y_te, bloquear_te, valor_te, p_te, n=n_bootstrap, seed=random_state)
    for nome, m in avaliacao.items():
        _log(_resumo(nome, m))
    m = avaliacao['politica_escolhida']
    _log(f'  PR-AUC {m["pr_auc"]:.4f} [{ic["pr_auc"][0]:.3f}, {ic["pr_auc"][1]:.3f}] | '
         f'recall [{ic["recall"][0]:.3f}, {ic["recall"][1]:.3f}] | precision [{ic["precision"][0]:.3f}, '
         f'{ic["precision"][1]:.3f}] | economia [{ic["economia"][0]:.1%}, {ic["economia"][1]:.1%}]')

    # --- 7. Importância por permutação (validação) ---
    _log('\n7. Importância por permutação (queda de PR-AUC na validação)')
    perm = permutation_importance(modelo, X_va, y_va, scoring='average_precision', n_repeats=5,
                                  random_state=random_state, n_jobs=1)
    importancia = (perm.importances_mean, perm.importances_std)
    ordem = np.argsort(importancia[0])[::-1]
    _log('  ' + ', '.join(f'{features[i]} ({importancia[0][i]:.3f})' for i in ordem[:5]))

    # --- 8. Explicação dos alertas de maior risco no teste (SHAP) ---
    _log('\n8. Explicação dos 10 alertas de maior risco no teste')
    artefato_avaliacao = {'pipeline': modelo, 'politica': politica, 'features': features,
                          'referencia': X_tr.sample(min(200, len(X_tr)), random_state=random_state)}
    alertas = teste.assign(prob_fraude=p_te)[bloquear_te].nlargest(10, 'prob_fraude')
    alertas = alertas[['Time', 'Amount', 'prob_fraude', ALVO]].join(explicar(artefato_avaliacao, alertas))
    alertas.to_csv(pasta_relatorios / 'alertas_exemplo.csv', index=False, float_format='%.4f')
    _log(alertas.head(3).to_string(index=False))

    # --- Gráficos ---
    _grafico_pr(y_te, p_te, m, politica, pasta_figuras / 'pr_curve.png')
    _grafico_custo(curva_va, limiar, custo_valor_va, usar_calibracao, pasta_figuras / 'custo_vs_limiar.png')
    _grafico_decisao(valor_te, p_te, y_te.to_numpy(), politica, pasta_figuras / 'decisao_no_teste.png')
    _grafico_matriz(y_te, bloquear_te, politica, pasta_figuras / 'confusion_matrix.png')
    _grafico_importancia(importancia, ordem[:15], features, pasta_figuras / 'feature_importance.png')

    # --- 9. Modelo de produção: mesmas escolhas, treinado com todos os dados ---
    _log(f'\n9. Modelo de produção treinado com todas as {len(df):,} transações')
    producao = novo_modelo().fit(df[features], df[ALVO])
    metadados = {'modelo': melhor, 'ajustado': bool(usar_ajustado), 'usa_hora': bool(usar_hora),
                 'calibrado': bool(usar_calibracao),
                 'transacoes_treino': len(df), 'time_max_treino': float(df['Time'].max())}
    salvar(pasta_modelos / 'modelo_fraude.joblib', producao, politica, features,
           df[features].sample(min(200, len(df)), random_state=random_state), metadados)

    def _numeros(d):
        return {k: (v if isinstance(v, (bool, str, type(None))) else
                    (int(v) if isinstance(v, (int, np.integer)) else float(v))) for k, v in d.items()}

    metricas = {
        'dados': {'transacoes': len(df), 'duplicadas_removidas': duplicadas,
                  'treino': len(treino), 'validacao': len(validacao), 'teste': len(teste),
                  'fraudes_teste': int(y_te.sum())},
        'custos': custos,
        'comparacao_cv': comparacao,
        'modelo': melhor,
        'ajustado': bool(usar_ajustado),
        'busca': {'melhores_parametros': _numeros(busca.best_params_), 'pr_auc_cv': busca.best_score_},
        'hora': {'usa': bool(usar_hora), 'pr_auc_cv_com_hora': notas_hora.mean(), 'pr_auc_cv_sem_hora': pr_auc_cv},
        'calibracao': calibracao,
        'validacao': {'limiar_unico': limiar, 'custo_limiar_unico': custo_limiar_va,
                      'custo_regra_por_valor': custo_valor_va},
        'politica': politica,
        'teste': avaliacao,
        'teste_ic95': ic,
        'importancia_permutacao': {features[i]: float(importancia[0][i]) for i in ordem},
        'producao': metadados,
        'segundos': round(time.time() - inicio),
    }
    (pasta_relatorios / 'metricas.json').write_text(json.dumps(metricas, indent=2, ensure_ascii=False, default=float),
                                                   encoding='utf-8')
    _log(f'\nConcluído em {metricas["segundos"]}s. Modelo em {pasta_modelos / "modelo_fraude.joblib"}')
    return metricas


def _salvar(fig, caminho):
    fig.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _grafico_pr(y, prob, m, politica, caminho):
    precisao, recall, _ = precision_recall_curve(y, prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precisao, color=AZUL, linewidth=2)
    ax.axhline(y.mean(), color=CINZA, linestyle='--', linewidth=1)
    ax.annotate(f'sem modelo ({y.mean():.2%} de fraudes)', xy=(0, y.mean()), xycoords=('axes fraction', 'data'),
                xytext=(4, 4), textcoords='offset points', ha='left', fontsize=8, color=TEXTO_SECUNDARIO)
    ax.plot(m['recall'], m['precision'], 'o', color=AZUL, markersize=8, markeredgecolor='white', markeredgewidth=2)
    ax.annotate(_nome_politica(politica), (m['recall'], m['precision']), xytext=(-10, -16),
                textcoords='offset points', ha='right', fontsize=9)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel('Recall (fraudes detectadas)')
    ax.set_ylabel('Precision (alertas que eram fraude)')
    ax.set_title(f'Curva Precision-Recall no teste (PR-AUC {m["pr_auc"]:.3f})', loc='left')
    _salvar(fig, caminho)


def _grafico_custo(curva, limiar, custo_valor, calibrada, caminho):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(curva['limiar'], curva['custo'], color=AZUL, linewidth=2)
    minimo = curva.loc[curva['limiar'] == limiar, 'custo'].iloc[0]
    ax.plot(limiar, minimo, 'o', color=AZUL, markersize=8, markeredgecolor='white', markeredgewidth=2)
    ax.annotate(f'melhor limiar único {limiar:.2f} (€{minimo:,.0f})', (limiar, minimo), xytext=(0, 12),
                textcoords='offset points', ha='center', fontsize=9)
    ax.axhline(custo_valor, color=LARANJA, linestyle='--', linewidth=1.5)
    ax.annotate(f'regra por valor (€{custo_valor:,.0f})', xy=(1, custo_valor), xycoords=('axes fraction', 'data'),
                xytext=(-4, -12), textcoords='offset points', ha='right', fontsize=9)
    ax.set_xlabel('Limiar único de decisão' + (' (probabilidade calibrada)' if calibrada else ''))
    ax.set_ylabel('Custo total (€)')
    ax.set_title('Custo das políticas na validação', loc='left')
    _salvar(fig, caminho)


def _grafico_decisao(valor, prob, y, politica, caminho):
    fig, ax = plt.subplots(figsize=(8, 5))
    prob = np.clip(prob, 1e-6, 1)
    ax.scatter(valor[y == 0], prob[y == 0], s=3, color=AZUL, alpha=0.25, linewidths=0, label='Legítimas')
    ax.scatter(valor[y == 1], prob[y == 1], s=22, color=LARANJA, edgecolors='white', linewidths=0.6, label='Fraudes')
    v = np.linspace(0, max(valor.max(), 10), 2000)
    por_valor = politica['custo_fp'] / (v + politica['custo_fn_fixo'] + politica['custo_fp'])
    if politica['tipo'] == 'valor':
        fronteira = por_valor
    else:
        fronteira = np.full_like(v, politica['limiar'])
        # Alternativa comparada na validação, para referência
        ax.plot(v, por_valor, color=LARANJA, linewidth=1.2, linestyle=':')
        ax.annotate('regra por valor (não escolhida)', (v[-1], por_valor[-1]), xytext=(-4, 6),
                    textcoords='offset points', ha='right', fontsize=8, color=TEXTO_SECUNDARIO)
    ax.plot(v, fronteira, color=TEXTO_SECUNDARIO, linewidth=1.5, linestyle='--')
    ax.annotate(f'{_nome_politica(politica)}: acima da linha, bloqueia', (v[-1], fronteira[-1]), xytext=(-4, -14),
                textcoords='offset points', ha='right', fontsize=9, color=TEXTO_SECUNDARIO)
    ax.set_xscale('symlog', linthresh=1)
    ax.set_yscale('log')
    ax.set_xlabel('Valor da transação (€, escala log)')
    ax.set_ylabel('Probabilidade de fraude (escala log)')
    ax.set_title(f'Decisão no teste: {_nome_politica(politica)}', loc='left')
    ax.legend(frameon=False, loc='lower right', markerscale=2)
    _salvar(fig, caminho)


def _grafico_matriz(y, bloquear, politica, caminho):
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ConfusionMatrixDisplay(confusion_matrix(y, bloquear), display_labels=['Legítima', 'Fraude']).plot(
        ax=ax, cmap='Blues', values_format=',d', colorbar=False)
    ax.grid(False)
    ax.set_xlabel('Previsto')
    ax.set_ylabel('Real')
    ax.set_title(f'Matriz de confusão no teste ({_nome_politica(politica)})', loc='left')
    _salvar(fig, caminho)


def _grafico_importancia(importancia, indices, features, caminho):
    media, desvio = importancia
    nomes = [features[i] for i in indices][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(nomes, media[indices][::-1], xerr=desvio[indices][::-1], color=AZUL, height=0.6,
            error_kw={'ecolor': TEXTO_SECUNDARIO, 'capsize': 2, 'elinewidth': 1})
    ax.set_xlabel('Queda da PR-AUC ao embaralhar a variável')
    ax.set_title('Importância por permutação (validação)', loc='left')
    ax.grid(axis='y', visible=False)
    _salvar(fig, caminho)
