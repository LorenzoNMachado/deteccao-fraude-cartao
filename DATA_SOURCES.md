# Dados e licenças de artefatos

## Credit Card Fraud Detection

O CSV completo não está versionado. O projeto usa o [dataset Credit Card Fraud Detection no Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
coletado em uma colaboração entre Worldline e o Machine Learning Group da ULB.
O cartão de metadados do Kaggle atualmente informa **“Database: Open Database,
Contents: Database Contents”**. Preserve a atribuição e as citações acadêmicas
indicadas na página de origem e confira os termos atuais antes de reutilizar ou
redistribuir o dataset.

`reports/alertas_exemplo.csv` contém linhas derivadas do conjunto de dados; os
gráficos e métricas também dependem dele. `models/modelo_fraude.joblib`, quando
gerado localmente, é derivado do mesmo material e não é versionado. Esses
artefatos não recebem direitos adicionais pela licença MIT, que cobre somente
o código original e a documentação de autoria do projeto.

O arquivo joblib é um formato baseado em pickle e pode executar código durante
a desserialização. Carregue apenas artefatos gerados por uma execução confiável
deste projeto; nunca aceite um modelo fornecido por usuário ou baixado de uma
origem não verificada.
