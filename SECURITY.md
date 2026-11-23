# Política de segurança

## Versões mantidas

Somente a branch `main` recebe manutenção. Não há versões publicadas ou
branches de release com suporte de segurança separado. O ambiente validado
pelo CI é Python 3.12.

## Como reportar uma vulnerabilidade

Use **Security → Advisories → Report a vulnerability** na página deste
repositório para enviar um relato privado. Não abra uma issue pública nem
publique detalhes de exploração antes de conversar em privado com o mantenedor.
Inclua o componente afetado, o impacto e passos mínimos para reproduzir; nunca
envie credenciais ou dados de cartão reais.

O projeto é mantido como portfólio e pesquisa; não há prazo de resposta ou
garantia de correção. Issues públicas podem ser usadas para problemas que não
sejam vulnerabilidades.

## Escopo e limites

Relatos relevantes incluem credenciais expostas, validação insuficiente da API,
execução de artefatos não confiáveis, dependências e workflows. O modelo
serializado por joblib deve ser carregado somente quando foi produzido por uma
origem confiável. O CSV de origem e os artefatos derivados têm direitos e
restrições separados da licença MIT do código; consulte
[`DATA_SOURCES.md`](DATA_SOURCES.md). Scanners não provam que o sistema está
seguro.
