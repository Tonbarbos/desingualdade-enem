# Desigualdade Social e Desempenho no ENEM — Espírito Santo

Painel interativo que analisa a relação entre desigualdade social e desempenho no ENEM 2024 nos 78 municípios do Espírito Santo, cruzando microdados do ENEM com indicadores socioeconômicos, gastos educacionais e qualidade do ensino.

🔗 **[Acessar o painel](https://desingualdade-enem.streamlit.app)**

---
## Objetivo

Este projeto tem como objetivo analisar como fatores socioeconômicos influenciam o desempenho educacional no ENEM, utilizando dados reais e técnicas estatísticas e de machine learning para gerar insights relevantes.

## O que o painel oferece

- **Explorador de Correlação** — scatter interativo entre qualquer par de indicadores com r de Pearson, R² e linha de tendência
- **Testes Estatísticos** — regressão linear simples, teste t por grupos e tabela completa de correlações
- **Outliers e Distribuição** — boxplot, ranking e estatísticas descritivas por indicador
- **Machine Learning** — K-Means (agrupamento de municípios por perfil socioeconômico), Random Forest (importância de variáveis) e previsão de notas para 2025
- **Evolução Temporal** — série histórica 2015–2024 com comparação por quartil de IDHM e análise do impacto da pandemia

## Fontes de dados

| Fonte | Dados |
|---|---|
| INEP — Microdados ENEM 2024 | Notas por município |
| Atlas Brasil / PNUD (Censo 2010) | IDHM, Gini, analfabetismo, escolaridade |
| CadÚnico 2024 (MDS) | Índice de Vulnerabilidade (IVCAD) |
| SIOPE 2023 (FNDE) | Gastos educacionais municipais |
| Censo Escolar 2023 (INEP) | IDEB, SAEB, rendimento escolar |

## Metodologia

- Análise exploratória de dados (EDA)
- Cálculo de correlações (Pearson)
- Modelos de regressão linear
- Algoritmos de Machine Learning (K-Means e Random Forest)
- Visualização interativa com Plotly

## Tecnologias

Python · Streamlit · Plotly · scikit-learn · pandas

## Integrantes do Grupo

- **[Ana Luiza Menelli Taylor](https://github.com/analuizataylor)**
- Danton Barbosa
