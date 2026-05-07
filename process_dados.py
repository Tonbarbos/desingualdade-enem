"""
process_duplo.py
Gera dois arquivos CSV:
  - dados_publico.csv  → apenas candidatos de escola pública
  - dados_todos.csv    → todos os candidatos

IMPORTANTE: filtra por SG_UF_PROVA == 'ES' (candidatos que fizeram a prova no ES)
e agrupa por NO_MUNICIPIO_PROVA (município onde fez a prova).
Isso captura muito mais candidatos do que filtrar por escola (SG_UF_ESC),
e é metodologicamente mais correto para cruzar com indicadores socioeconômicos
municipais, pois representa quem estava vivendo naquele município no momento do exame.

Fontes de dados:
  - ENEM 2024: microdados INEP
  - Atlas Brasil: indicadores socioeconômicos Censo 2010
  - SIOPE 2023: gastos educacionais municipais (FNDE)
  - CadÚnico 2024: IVCAD e subíndices de vulnerabilidade (Observatório MDS)
    Download: paineis.mds.gov.br → Tabela → Exportar → Famílias_e_pessoas_por_municipio_ES.xlsx
"""

import pandas as pd
import unicodedata
from scipy import stats

ENEM_PATH    = r"microdados_enem_2024\DADOS\RESULTADOS_2024.csv"
ATLAS_PATH   = "base correta.xlsx"
EDU_PATH     = "municipios-educacao.csv"
CADUNICO_PATH= "Familias_e_pessoas_por_municipio_ES.xlsx"  # exportado do Observatório CadÚnico
UF_ALVO      = "ES"

def normalize(s):
    if pd.isna(s): return ""
    s = str(s).upper().strip()
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

# ── ATLAS BRASIL ──────────────────────────────────────────────────
def load_atlas():
    df = pd.read_excel(ATLAS_PATH)
    df = df[df['Territorialidades'].astype(str).str.endswith(' (ES)')].copy()
    df['Nome_Municipio'] = df['Territorialidades'].str.replace(' (ES)', '', regex=False)
    df['municipio_norm'] = df['Nome_Municipio'].apply(normalize)

    col_map = {
        'Índice de Gini 2010':                                                                              'GINI',
        'Expectativa de anos de estudo aos 18 anos de idade 2010':                                          'E_ANOSESTUDO',
        '% de 25 anos ou mais de idade com ensino fundamental completo 2010':                               'PERC_FUND_COMP',
        '% de 25 anos ou mais de idade com ensino médio completo 2010':                                     'PERC_MED_COMP',
        '% de 15 a 17 anos de idade na escola 2010':                                                        'T_FREQ1517_FUND',
        'Taxa de analfabetismo - 15 anos ou mais de idade 2010':                                            'TX_ANALF',
        'IDHM Renda 2010':                                                                                  'IDHM_R',
        'IDHM Educação 2010':                                                                               'IDHM_E',
        'IDHM 2010':                                                                                        'IDHM',
        '% de crianças que vivem em domicílios em que nenhum dos moradores tem o ensino fundamental completo 2010': 'CRIAN_VULN',
        '% de 15 a 24 anos de idade que não estudam nem trabalham em domicílios vulneráveis à pobreza 2010': 'NEET_VULN',
    }
    df = df.rename(columns=col_map)
    keep = ['municipio_norm', 'Nome_Municipio', 'GINI', 'E_ANOSESTUDO',
            'PERC_FUND_COMP', 'PERC_MED_COMP', 'T_FREQ1517_FUND', 'TX_ANALF',
            'IDHM_R', 'IDHM_E', 'IDHM', 'CRIAN_VULN', 'NEET_VULN']
    df = df[[c for c in keep if c in df.columns]]
    for c in df.select_dtypes('object').columns:
        if c not in ['municipio_norm', 'Nome_Municipio']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# ── SIOPE ─────────────────────────────────────────────────────────
def load_edu():
    df = pd.read_csv(EDU_PATH, encoding='latin-1', sep=';', decimal=',')
    df = df[df['Ano'] == 2023].copy()
    df['municipio_norm'] = df['Municipio'].apply(normalize)
    df = df.rename(columns={
        'Aplicacao':           'EDU_Aplicacao_Total',
        'AplicacaoPercentual': 'EDU_Perc_Aplicacao',
        'Alunos':              'EDU_Alunos',
        'AplicacaoPorAluno':   'EDU_Investimento_Aluno',
        'ReceitaFUNDEB':       'EDU_ReceitaFUNDEB',
    })
    return df[['municipio_norm', 'EDU_Aplicacao_Total', 'EDU_Perc_Aplicacao',
               'EDU_Alunos', 'EDU_Investimento_Aluno', 'EDU_ReceitaFUNDEB']]

# ── ENEM ──────────────────────────────────────────────────────────
def load_enem():
    print("  Lendo microdados ENEM (pode demorar)...")
    cols = [
        'SG_UF_PROVA',           # UF onde fez a prova — usado para filtrar ES
        'NO_MUNICIPIO_PROVA',     # Município onde fez a prova — usado para agrupar
        'TP_DEPENDENCIA_ADM_ESC', # Tipo de escola onde estudou o ensino médio
        'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO',
    ]
    chunks = []
    for chunk in pd.read_csv(ENEM_PATH, encoding='latin1', sep=';',
                             chunksize=500_000, usecols=cols):
        # Filtra candidatos que fizeram a prova no ES
        chunk = chunk[chunk['SG_UF_PROVA'] == UF_ALVO].dropna(subset=['NO_MUNICIPIO_PROVA'])
        if not chunk.empty:
            chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    notas = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
    df['NOTA_MEDIA'] = df[notas].mean(axis=1)
    print(f"  Total: {len(df):,} candidatos que fizeram prova no ES")
    return df

def agregar(df_enem):
    agg = df_enem.groupby('NO_MUNICIPIO_PROVA').agg(
        NU_NOTA_CN      = ('NU_NOTA_CN',      'mean'),
        NU_NOTA_CH      = ('NU_NOTA_CH',      'mean'),
        NU_NOTA_LC      = ('NU_NOTA_LC',      'mean'),
        NU_NOTA_MT      = ('NU_NOTA_MT',      'mean'),
        NU_NOTA_REDACAO = ('NU_NOTA_REDACAO', 'mean'),
        NOTA_MEDIA      = ('NOTA_MEDIA',      'mean'),
        QTD_CANDIDATOS  = ('NO_MUNICIPIO_PROVA', 'count'),
        PERC_ESCOLA_PUB = ('TP_DEPENDENCIA_ADM_ESC',
                           lambda x: (x.isin([1, 2, 3])).mean() * 100),
    ).reset_index()
    agg['municipio_norm'] = agg['NO_MUNICIPIO_PROVA'].apply(normalize)
    return agg

def load_censo_escolar():
    """
    Lê a planilha do INEP com IDEB, Nota SAEB e Indicador de Rendimento
    do ensino médio por município do ES (2023).
    Download: inep.gov.br/educacao-basica/ideb/resultados
    """
    filepath = "divulgacao_ensino_medio_municipios_2023.ods"
    df = pd.read_excel(filepath, engine='odf', header=8)
    df = df.rename(columns={
        df.columns[0]:  'SG_UF',
        df.columns[2]:  'NO_MUNICIPIO',
        df.columns[3]:  'REDE',
        df.columns[27]: 'IND_REND_2023',
        df.columns[38]: 'NOTA_MEDIA_SAEB_2023',
        df.columns[43]: 'IDEB_2023',
    })
    df = df[df['SG_UF'] == 'ES'].copy()
    df = df[df['REDE'] == 'Pública'].copy()
    df['municipio_norm'] = df['NO_MUNICIPIO'].apply(normalize)
    for col in ['IND_REND_2023', 'NOTA_MEDIA_SAEB_2023', 'IDEB_2023']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"  Censo Escolar: {len(df)} municípios ES (rede pública)")
    return df[['municipio_norm', 'IND_REND_2023', 'NOTA_MEDIA_SAEB_2023', 'IDEB_2023']]

# ── CADUNICO ──────────────────────────────────────────────────────
def load_cadunico():
    """
    Lê a planilha exportada do Observatório CadÚnico com IVCAD por município.
    Download: paineis.mds.gov.br → Tabela → Exportar
    """
    df = pd.read_excel(CADUNICO_PATH)
    df = df.rename(columns={
        'Cód. IBGE': 'cod_ibge',
        'IVCAD': 'IVCAD',
        'NC':  'IVCAD_NC',
        'DPI': 'IVCAD_DPI',
        'DCA': 'IVCAD_DCA',
        'TQA': 'IVCAD_TQA',
        'DR':  'IVCAD_DR',
        'CH':  'IVCAD_CH',
        '% relativo ao Censo 2022': 'IVCAD_COBERTURA',
    })
    df['municipio_norm'] = df['Município'].apply(normalize)
    cols = ['municipio_norm', 'IVCAD', 'IVCAD_NC', 'IVCAD_DPI', 'IVCAD_DCA',
            'IVCAD_TQA', 'IVCAD_DR', 'IVCAD_CH', 'IVCAD_COBERTURA']
    print(f"  CadÚnico: {len(df)} municípios")
    return df[cols]

def gerar_csv(df_atlas, df_edu, df_cad, df_inep, df_enem, output_path, label):
    agg = agregar(df_enem)
    df  = df_atlas.merge(df_edu,  on='municipio_norm', how='left')
    df  = df.merge(df_cad,  on='municipio_norm', how='left')
    df  = df.merge(df_inep, on='municipio_norm', how='left')
    df  = df.merge(agg, on='municipio_norm', how='left')
    df.to_csv(output_path, index=False)
    n_com_nota = df['NOTA_MEDIA'].notna().sum()
    print(f"  Salvo: {output_path}  ({n_com_nota} municípios com nota | {int(df['QTD_CANDIDATOS'].sum()):,} candidatos)")

    print(f"  Correlações com NOTA_MEDIA [{label}]:")
    for col in ['IDHM','IDHM_R','GINI','TX_ANALF','NEET_VULN',
                'IVCAD_DR','IVCAD_TQA',
                'NOTA_MEDIA_SAEB_2023','IDEB_2023','IND_REND_2023',
                'EDU_Perc_Aplicacao','EDU_Investimento_Aluno']:
        sub = df[['NOTA_MEDIA', col]].dropna()
        if len(sub) < 5: continue
        r, p = stats.pearsonr(sub['NOTA_MEDIA'].astype(float), sub[col].astype(float))
        print(f"    {'✓' if p < 0.05 else '✗'} r={r:+.3f}  p={p:.3f}  n={len(sub)}  {col}")

if __name__ == '__main__':
    print("Carregando Atlas Brasil...")
    df_atlas = load_atlas()

    print("Carregando SIOPE 2023...")
    df_edu = load_edu()

    print("Carregando CadÚnico 2024...")
    df_cad = load_cadunico()

    print("Carregando Censo Escolar 2023 (INEP)...")
    df_inep = load_censo_escolar()

    print("Carregando ENEM 2024...")
    df_enem_todos = load_enem()

    print("\n[1/3] Gerando dados_todos.csv (pública + privada)...")
    gerar_csv(df_atlas, df_edu, df_cad, df_inep, df_enem_todos, 'dados_todos.csv', 'todos')

    print("\n[2/3] Gerando dados_publico.csv (apenas escola pública)...")
    df_enem_pub = df_enem_todos[
        df_enem_todos['TP_DEPENDENCIA_ADM_ESC'].isin([1, 2, 3])
    ].copy()
    gerar_csv(df_atlas, df_edu, df_cad, df_inep, df_enem_pub, 'dados_publico.csv', 'pública')

    print("\n[3/3] Gerando dados_privado.csv (apenas escola privada)...")
    df_enem_priv = df_enem_todos[
        df_enem_todos['TP_DEPENDENCIA_ADM_ESC'] == 4
    ].copy()
    gerar_csv(df_atlas, df_edu, df_cad, df_inep, df_enem_priv, 'dados_privado.csv', 'privada')

    print("\nPronto! Suba os CSVs para o GitHub.")