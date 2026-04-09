import pdfplumber
import os
import re

def extract_siope_data(filepath):
    municipio = None
    investimento_aluno = None
    perc_impostos = None
    
    with pdfplumber.open(filepath) as pdf:
        text = ""
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
                
    m = re.search(r'Munic[i|í|]pio:\s*(.+)', text)
    if m:
        municipio = m.group(1).strip()
        
    print(f"--- {filepath} ---")
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if "1.1 " in line and "Percentual de" in line:
            # print surrounding lines
            print("1.1 contextual block:", lines[i:i+4])
        if "4.9 " in line and "educacional por" in line:
            print("4.9 contextual block:", lines[i-1:i+3])

for f in os.listdir(r"e:\Ollama\DesingualdadeEnem"):
    if f.startswith("SIOPE-VITORIA") or f.startswith("SIOPE.VILAVELHA"):
        extract_siope_data(os.path.join(r"e:\Ollama\DesingualdadeEnem", f))
