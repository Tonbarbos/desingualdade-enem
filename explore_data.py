import pandas as pd
import pdfplumber

def search_cols(filepath):
    print("--- Searching cols in Excel ---")
    df = pd.read_excel(filepath, nrows=1)
    cols = df.columns.tolist()
    match_cols = [c for c in cols if 'cod' in c.lower() or 'mun' in c.lower() or 'ibge' in c.lower()]
    print("Matched columns:", match_cols)
    print("First row values for these cols:")
    for c in match_cols:
        print(f"  {c}: {df[c].iloc[0]}")

def check_pdf(filepath):
    print(f"\n--- {filepath} ---")
    with pdfplumber.open(filepath) as pdf:
        text = ""
        for page in pdf.pages[:2]:
            text += page.extract_text() + "\n"
        print("First 1500 chars of PDF:")
        print(text[:1500])

if __name__ == "__main__":
    search_cols(r"e:\Ollama\DesingualdadeEnem\Es-Desingualdade(censo2010-+PNAD2012-2021).xlsx")
    check_pdf(r"e:\Ollama\DesingualdadeEnem\SIOPE-VITORIA.pdf")
