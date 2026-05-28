"""
process_temporal.py — Gera dados_temporal.csv com nota média por município/ano (2012-2024)
Uso: python process_temporal.py
"""
import os
import unicodedata
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

NOTE_COLS = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']

YEAR_FILES = {
    2012: 'microdados_enem_2012/DADOS/MICRODADOS_ENEM_2012.csv',
    2013: 'microdados_enem_2013/DADOS/MICRODADOS_ENEM_2013.csv',
    2014: 'microdados_enem_2014/DADOS/MICRODADOS_ENEM_2014.csv',
    2015: 'microdados_enem_2015/DADOS/MICRODADOS_ENEM_2015.csv',
    2016: 'microdados_enem_2016/DADOS/microdados_enem_2016.csv',
    2017: 'microdados_enem_2017/DADOS/MICRODADOS_ENEM_2017.csv',
    2018: 'microdados_enem_2018/DADOS/MICRODADOS_ENEM_2018.csv',
    2019: 'microdados_enem_2019/DADOS/MICRODADOS_ENEM_2019.csv',
    2020: 'microdados_enem_2020/DADOS/MICRODADOS_ENEM_2020.csv',
    2021: 'microdados_enem_2021/DADOS/MICRODADOS_ENEM_2021.csv',
    2022: 'microdados_enem_2022/DADOS/MICRODADOS_ENEM_2022.csv',
    2023: 'microdados_enem_2023/DADOS/MICRODADOS_ENEM_2023.csv',
    2024: 'microdados_enem_2024/DADOS/RESULTADOS_2024.csv',
}


def normalize(s):
    if pd.isna(s):
        return ''
    s = str(s).upper().strip()
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def process_year(year, rel_path):
    path = os.path.join(BASE, rel_path)
    if not os.path.exists(path):
        print(f'  {year}: arquivo não encontrado — {rel_path}')
        return None

    cols_needed = ['SG_UF_PROVA', 'NO_MUNICIPIO_PROVA'] + NOTE_COLS
    print(f'  {year}: lendo {os.path.basename(path)} ...', end=' ', flush=True)

    df = pd.read_csv(path, sep=';', encoding='latin-1',
                     usecols=cols_needed, low_memory=True)

    df_es = df[df['SG_UF_PROVA'] == 'ES'].copy()

    for col in NOTE_COLS:
        df_es[col] = pd.to_numeric(df_es[col], errors='coerce')
        # Zeros indicam ausência em alguns anos (não são notas válidas no TRI)
        df_es[col] = df_es[col].replace(0, pd.NA)

    # Nota média = média das 5 provas (ignora NaN por prova individual)
    df_es['NOTA_MEDIA'] = df_es[NOTE_COLS].mean(axis=1)

    # Manter só candidatos com pelo menos 3 notas válidas (presença razoável)
    df_es = df_es[df_es[NOTE_COLS].notna().sum(axis=1) >= 3]

    agg = df_es.groupby('NO_MUNICIPIO_PROVA').agg(
        NOTA_MEDIA=('NOTA_MEDIA', 'mean'),
        NU_NOTA_MT=('NU_NOTA_MT', 'mean'),
        NU_NOTA_LC=('NU_NOTA_LC', 'mean'),
        NU_NOTA_CH=('NU_NOTA_CH', 'mean'),
        NU_NOTA_CN=('NU_NOTA_CN', 'mean'),
        NU_NOTA_REDACAO=('NU_NOTA_REDACAO', 'mean'),
        QTD_CANDIDATOS=('NOTA_MEDIA', 'count'),
    ).reset_index()

    agg['Ano'] = year
    agg['municipio_norm'] = agg['NO_MUNICIPIO_PROVA'].apply(normalize)

    print(f'{len(agg)} municípios | {int(agg["QTD_CANDIDATOS"].sum())} candidatos')
    return agg


if __name__ == '__main__':
    print('=== Processando microdados históricos do ENEM (ES) ===\n')

    frames = []
    for year in sorted(YEAR_FILES):
        result = process_year(year, YEAR_FILES[year])
        if result is not None:
            frames.append(result)

    if not frames:
        print('Nenhum arquivo encontrado.')
    else:
        df_out = pd.concat(frames, ignore_index=True)

        # Arredondar notas
        note_cols_out = ['NOTA_MEDIA', 'NU_NOTA_MT', 'NU_NOTA_LC',
                         'NU_NOTA_CH', 'NU_NOTA_CN', 'NU_NOTA_REDACAO']
        df_out[note_cols_out] = df_out[note_cols_out].round(2)

        out_path = os.path.join(BASE, 'dados_temporal.csv')
        df_out.to_csv(out_path, index=False, encoding='utf-8')

        print(f'\n=== Concluído ===')
        print(f'Arquivo: dados_temporal.csv')
        print(f'Linhas:  {len(df_out)} (município × ano)')
        print(f'Anos:    {sorted(df_out["Ano"].unique())}')
        print(f'Municípios únicos: {df_out["municipio_norm"].nunique()}')
