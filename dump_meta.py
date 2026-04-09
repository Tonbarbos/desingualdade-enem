import pandas as pd
import pdfplumber
import json

def check_excel_cols(filepath):
    df = pd.read_excel(filepath, nrows=1)
    cols = df.columns.tolist()
    with open("excel_cols.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(cols))
    
def check_pdf_all(filepath):
    with pdfplumber.open(filepath) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        with open("siope_text.txt", "w", encoding="utf-8") as f:
            f.write(text)

if __name__ == "__main__":
    check_excel_cols(r"e:\Ollama\DesingualdadeEnem\Es-Desingualdade(censo2010-+PNAD2012-2021).xlsx")
    check_pdf_all(r"e:\Ollama\DesingualdadeEnem\SIOPE-ES-2022.pdf")
