import os
import re
import pandas as pd
import unicodedata

def normalize_string(s):
    if pd.isna(s) or s is None:
        return ""
    s = str(s).upper().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s

def parse_brl(valor_str):
    """Converte string 'R$1.234,56' ou '1.234,56' para float."""
    v = valor_str.replace('R$', '').replace('.', '').replace(',', '.').strip()
    try:
        return float(v)
    except ValueError:
        return None

def parse_percent(valor_str):
    """Converte string '28,88%' para float (28.88)."""
    v = valor_str.replace('%', '').replace(',', '.').strip()
    try:
        return float(v)
    except ValueError:
        return None

def extract_last_bimestre(line_text):
    """
    Dado um trecho de texto contendo vários valores bimestrais R$X,XX,
    retorna o ÚLTIMO valor (6º bimestre = acumulado anual).
    """
    # Encontra todos os R$valor no texto
    values = re.findall(r'R\$[\d.,]+', line_text)
    if values:
        return parse_brl(values[-1])
    return None

def extract_last_percent(line_text):
    """
    Dado um trecho de texto com percentuais, retorna o último (6º bimestre).
    """
    values = re.findall(r'[\d.,]+%', line_text)
    if values:
        return parse_percent(values[-1])
    return None

def load_municipios_educacao(ano=2023):
    """
    Lê o arquivo municipios-educacao.csv (fonte SIOPE/FNDE) e retorna
    os dados do ano solicitado mapeados para as colunas do dashboard.
    """
    filepath = r"e:\Ollama\DesingualdadeEnem\municipios-educacao.csv"
    df = pd.read_csv(filepath, encoding='latin-1', sep=';', decimal=',')
    df_ano = df[df['Ano'] == ano].copy()

    df_ano['municipio_norm'] = df_ano['Municipio'].apply(normalize_string)

    # Mapear colunas do CSV para os nomes usados no dashboard
    df_out = df_ano[['municipio_norm']].copy()
    df_out['EDU_Investimento_Aluno']   = df_ano['AplicacaoPorAluno']
    df_out['EDU_Perc_Aplicacao']       = df_ano['AplicacaoPercentual']
    df_out['EDU_Alunos']               = df_ano['Alunos']
    df_out['EDU_Aplicacao_Total']      = df_ano['Aplicacao']

    print(f"  municipios-educacao.csv ({ano}): {len(df_out)} municípios carregados")
    return df_out


def process_excel():
    filepath = r"e:\Ollama\DesingualdadeEnem\base correta.xlsx"
    df = pd.read_excel(filepath)
    # Filtrar apenas municípios do ES (excluir Brasil e Espírito Santo total)
    df = df[df['Territorialidades'].astype(str).str.endswith(' (ES)')].copy()
    df['Nome_Municipio'] = df['Territorialidades'].str.replace(' (ES)', '', regex=False)
    df['municipio_norm'] = df['Nome_Municipio'].apply(normalize_string)

    # Mapeamento completo das colunas da nova base correta
    col_map = {
        # Renda
        'Renda per capita 2010':                                                                       'RDPC',
        'Índice de Gini 2010':                                                                         'GINI',
        # Educação
        'Expectativa de anos de estudo aos 18 anos de idade 2010':                                     'E_ANOSESTUDO',
        '% de 25 anos ou mais de idade com ensino fundamental completo 2010':                          'PERC_FUND_COMP',
        '% de 25 anos ou mais de idade com ensino médio completo 2010':                                'PERC_MED_COMP',
        '% de 15 a 17 anos de idade na escola 2010':                                                   'T_FREQ1517',
        'Taxa de analfabetismo - 15 anos ou mais de idade 2010':                                       'TX_ANALF',
        # Desenvolvimento Humano
        'IDHM Renda 2010':                                                                             'IDHM_R',
        'IDHM Educação 2010':                                                                          'IDHM_E',
        'IDHM 2010':                                                                                   'IDHM',
        # Vulnerabilidade
        '% de crianças que vivem em domicílios em que nenhum dos moradores tem o ensino fundamental completo 2010': 'CRIAN_VULN',
        '% de 15 a 24 anos de idade que não estudam nem trabalham em domicílios vulneráveis à pobreza 2010': 'NEET_VULN',
    }

    keep_cols = ['municipio_norm', 'Nome_Municipio'] + list(col_map.keys())
    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols]
    df = df.rename(columns=col_map)
    return df


def process_enem():
    print("Lendo microdados ENEM 2024 (agrupando escolas)...")
    resultados_es = []
    chunk_iter = pd.read_csv(
        r"e:\Ollama\DesingualdadeEnem\microdados_enem_2024\DADOS\RESULTADOS_2024.csv",
        encoding='latin1', sep=';', chunksize=1000000,
        usecols=['SG_UF_ESC', 'NO_MUNICIPIO_ESC', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
    )
    for chunk in chunk_iter:
        chunk = chunk[chunk['SG_UF_ESC'] == 'ES']
        chunk = chunk.dropna(subset=['NO_MUNICIPIO_ESC'])
        resultados_es.append(chunk)
    df_res = pd.concat(resultados_es, ignore_index=True)

    print("Calculando médias por município...")
    df_res['NOTA_MEDIA'] = df_res[['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']].mean(axis=1)

    df_agg = df_res.groupby('NO_MUNICIPIO_ESC').agg(
        NU_NOTA_CN=('NU_NOTA_CN', 'mean'),
        NU_NOTA_CH=('NU_NOTA_CH', 'mean'),
        NU_NOTA_LC=('NU_NOTA_LC', 'mean'),
        NU_NOTA_MT=('NU_NOTA_MT', 'mean'),
        NU_NOTA_REDACAO=('NU_NOTA_REDACAO', 'mean'),
        NOTA_MEDIA=('NOTA_MEDIA', 'mean'),
        QTD_CANDIDATOS=('NO_MUNICIPIO_ESC', 'count')
    ).reset_index()

    df_agg['municipio_norm'] = df_agg['NO_MUNICIPIO_ESC'].apply(normalize_string)
    return df_agg


if __name__ == "__main__":
    print("=" * 60)
    print("1. Carregando dados educacionais (municipios-educacao.csv)...")
    print("=" * 60)
    df_edu = load_municipios_educacao(ano=2023)
    print(df_edu[['municipio_norm', 'EDU_Investimento_Aluno', 'EDU_Perc_Aplicacao']].to_string())

    print("\n" + "=" * 60)
    print("2. Processando Atlas Brasil (Desigualdade)...")
    print("=" * 60)
    df_excel = process_excel()
    print(f"Excel: {len(df_excel)} municípios")

    print("\n" + "=" * 60)
    print("3. Processando ENEM 2024...")
    print("=" * 60)
    df_enem = process_enem()
    print(f"ENEM: {len(df_enem)} municípios")

    print("\n" + "=" * 60)
    print("4. Unindo os datasets...")
    print("=" * 60)
    df_merged = df_excel.merge(df_edu, on='municipio_norm', how='left')
    df_final  = df_merged.merge(df_enem, on='municipio_norm', how='left')

    df_final.to_csv(r"e:\Ollama\DesingualdadeEnem\dados_compilados.csv", index=False)
    print(f"\nSUCESSO! Salvo como dados_compilados.csv com {len(df_final)} municípios.")

    # Verificação rápida dos valores educacionais
    print("\n--- Verificação Gastos Educacionais (municípios com dados) ---")
    edu_cols = [c for c in df_final.columns if 'EDU_' in c]
    print(df_final[['Nome_Municipio'] + edu_cols].dropna(subset=['EDU_Investimento_Aluno']).to_string())
