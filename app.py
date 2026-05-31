import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

st.set_page_config(page_title="Desigualdade vs ENEM no ES", page_icon=":material/school:", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');
            
    .material-symbols-outlined {
        font-family: 'Material Symbols Outlined' !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-size: 24px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-feature-settings: 'liga' !important;
        -webkit-font-smoothing: antialiased;
    }

    /* ── Base ── */
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background-color: #080D14; }

    /* ── Tipografia global ── */
    h1 { color: #F1F5F9; font-weight: 700; letter-spacing: -0.5px; }
    h2, h3, h4 { color: #E2E8F0; font-weight: 600; }
    p, li { color: #94A3B8; line-height: 1.7; }

    /* ── Cabeçalho do app ── */
    .app-header {
        background: linear-gradient(135deg, rgba(14,22,36,0.95) 0%, rgba(15,28,50,0.95) 100%);
        border: 1px solid rgba(56,189,248,0.15);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .app-header::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #0EA5E9, #38BDF8, #7DD3FC, #38BDF8, #0EA5E9);
        background-size: 200% 100%;
        animation: shimmer 3s linear infinite;
    }
    @keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
    .app-header h1 { margin: 0 0 8px 0; font-size: 1.6rem; }
    .app-header p  { margin: 0; font-size: 0.92rem; color: #64748B; }

    /* ── Cards de métricas ── */
    .metric-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.9), rgba(20,30,50,0.9));
        backdrop-filter: blur(12px);
        border-radius: 14px;
        padding: 22px 20px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
        margin-bottom: 16px;
        border: 1px solid rgba(255,255,255,0.07);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card::after {
        content: '';
        position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, rgba(56,189,248,0.4), transparent);
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.08);
        border-color: rgba(56,189,248,0.2);
    }
    .metric-icon  { font-size: 1.4rem; margin-bottom: 10px; opacity: 0.85; }
    .metric-title { color: #64748B; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
    .metric-value { color: #38BDF8; font-size: 2rem; font-weight: 700; margin-top: 6px; font-family: 'DM Mono', monospace; letter-spacing: -1px; }
    .metric-sub   { color: #475569; font-size: 0.78rem; margin-top: 4px; }

    /* ── Caixas de estatísticas ── */
    .stat-box {
        background: rgba(15,23,42,0.6);
        border-radius: 10px;
        padding: 14px 16px;
        border: 1px solid rgba(255,255,255,0.06);
        margin-top: 8px;
        transition: border-color 0.2s;
    }
    .stat-box:hover { border-color: rgba(56,189,248,0.2); }

    /* ── Banners de alerta / info ── */
    .warn-banner {
        background: rgba(234,179,8,0.08);
        border: 1px solid rgba(234,179,8,0.25);
        border-left: 3px solid rgba(234,179,8,0.7);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        color: #A3A39E;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    .info-banner {
        background: rgba(99,102,241,0.07);
        border: 1px solid rgba(99,102,241,0.2);
        border-left: 3px solid rgba(99,102,241,0.6);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        color: #A3A3B8;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    .filter-banner {
        background: rgba(56,189,248,0.07);
        border: 1px solid rgba(56,189,248,0.2);
        border-left: 3px solid rgba(56,189,248,0.6);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        color: #94A3B8;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    /* ── Separadores de seção ── */
    .section-divider {
        display: flex; align-items: center; gap: 12px;
        margin: 28px 0 20px 0;
    }
    .section-divider-line {
        flex: 1; height: 1px;
        background: linear-gradient(90deg, rgba(56,189,248,0.3), transparent);
    }

    /* ── Filtros ── */
    .filter-section {
        background: rgba(15,23,42,0.6);
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 20px;
    }

    /* ── Rótulos dos inputs ── */
    .input-label {
        color: #64748B;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 4px;
    }

    /* ── Streamlit overrides ── */
    .stRadio > label { color: #94A3B8 !important; font-size: 0.85rem !important; }
    div[data-testid="stMetricValue"] { font-family: 'DM Mono', monospace !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: rgba(15,23,42,0.8); border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 16px; color: #64748B; font-size: 0.88rem; }
    .stTabs [aria-selected="true"] { background: rgba(56,189,248,0.15) !important; color: #38BDF8 !important; font-weight: 600; }
    [data-testid="stSidebar"] { background: rgba(10,18,30,0.95); }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    div[data-testid="stExpander"] { border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 10px !important; }
    .streamlit-expanderHeader { font-size: 0.9rem !important; }
    hr { border-color: rgba(255,255,255,0.06) !important; margin: 24px 0 !important; }

    /* ── Scroll customizado ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(15,23,42,0.5); }
    ::-webkit-scrollbar-thumb { background: rgba(56,189,248,0.3); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(56,189,248,0.5); }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE — preserva seleções ao trocar filtros ───────────
for key, default in {
    'tipo_escola': "Pública + Privada (todos)",
    'min_cand':    30,
    'cat_x':       'NOTAS DO ENEM',
    'cat_y':       'DESENVOLVIMENTO HUMANO (Censo 2010)',
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── FILTROS GLOBAIS ───────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>Desigualdade Social e Desempenho no ENEM — Espírito Santo 2024</h1>
    <p>Análise exploratória das notas do ENEM 2024 nos municípios do ES, cruzadas com indicadores socioeconômicos
    (Atlas Brasil / Censo 2010) e gastos educacionais municipais (SIOPE 2023).</p>
</div>
""", unsafe_allow_html=True)

tipo_escola = st.radio(
    ":material/school: Candidatos incluídos:",
    ["Apenas escola pública", "Pública + Privada (todos)"],
    horizontal=True,
    key="tipo_escola",
    help=(
        "📌 Apenas escola pública (recomendado para análise de correlação):\n"
        "Inclui somente candidatos identificados como concluintes de 2024 "
        "da rede pública (Federal, Estadual ou Municipal), cruzados com o "
        "Censo Escolar 2024 pelo CPF. São ~25 mil candidatos no ES.\n\n"
        "⚠️ Pública + Privada (todos os candidatos presentes):\n"
        "Inclui os ~74 mil candidatos que fizeram a prova no ES, sendo:\n"
        "• ~25 mil concluintes de escola pública em 2024\n"
        "• ~3,5 mil concluintes de escola privada em 2024\n"
        "• ~45 mil egressos de anos anteriores (já formados, sem vínculo "
        "escolar atual — TP_DEPENDENCIA_ADM_ESC em branco)\n\n"
        "Nota: a opção 'Apenas escola privada' foi removida porque os "
        "~3.500 candidatos privados distribuídos por 78 municípios resultam "
        "em amostras municipais insuficientes (média de 45 por município), "
        "tornando as correlações estatisticamente inválidas."
    ),
)

arquivo = "dados_publico.csv" if tipo_escola == "Apenas escola pública" else "dados_todos.csv"

 # FIX 7: leitura segura dos arquivos CSV com tratamento de erros de importação
@st.cache_data
def load_csv_data(path):
    try:
        dados = pd.read_csv(path)

        if dados.empty:
            return None, f"O arquivo **{path}** foi encontrado, mas está vazio."

        return dados, None

    except FileNotFoundError:
        return None, f"Arquivo **{path}** não encontrado."

    except pd.errors.EmptyDataError:
        return None, f"O arquivo **{path}** está vazio ou não possui dados válidos."

    except pd.errors.ParserError:
        return None, f"Não foi possível interpretar o arquivo **{path}**. Verifique se o CSV está bem formatado."

    except Exception as erro:
        return None, f"Erro inesperado ao carregar o arquivo **{path}**: {erro}"


df_raw, erro_dados = load_csv_data(arquivo)

if erro_dados:
    st.error(erro_dados)
    st.info("Execute `process_duplo.py` para gerar os arquivos `dados_publico.csv` e `dados_todos.csv`.")
    st.stop()

st.markdown("---")
min_cand = st.slider(
    ":material/group: Mínimo de candidatos por município:",
    min_value=10, max_value=300, step=10,
    key="min_cand",
    help="Municípios com poucos candidatos têm nota média instável e podem distorcer correlações."
)

df = df_raw[df_raw['QTD_CANDIDATOS'] >= min_cand].copy().reset_index(drop=True)
excluidos = len(df_raw) - len(df)

if excluidos > 0:
    st.markdown(
        f'<div class="warn-banner"><span class="material-symbols-outlined" style="vertical-align: middle; font-size: 1.1em;">warning</span> <b>{excluidos} município(s) excluído(s)</b> por ter menos de {min_cand} candidatos. '
        f'Restam <b>{len(df)} municípios</b> na análise. '
        f'<b>Atenção:</b> filtros muito altos podem deixar poucos municípios e tornar correlações instáveis.</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ── DICIONÁRIO DE INDICADORES — sem categoria PARTICIPAÇÃO ────────
INDICATORS_DICT = {
    'NOTAS DO ENEM': {
        'Nota Média Geral':  'NOTA_MEDIA',
        'Nota Redação':      'NU_NOTA_REDACAO',
        'Nota Matemática':   'NU_NOTA_MT',
        'Nota Linguagens':   'NU_NOTA_LC',
        'Nota Humanas':      'NU_NOTA_CH',
        'Nota Natureza':     'NU_NOTA_CN',
    },
    'VULNERABILIDADE SOCIAL (Censo 2010)': {
        'Índice de Gini':                          'GINI',
        'Taxa de Analfabetismo (15+ anos)':        'TX_ANALF',
        '% Crianças em Dom. Sem Ens. Fundamental': 'CRIAN_VULN',
        '% 15-24 anos Nem Estuda Nem Trabalha':    'NEET_VULN',
        '% de 15-17 ainda no Ensino Fundamental':  'T_FREQ1517_FUND',
    },
    'IVCAD — Vulnerabilidade CadÚnico (2024)': {
        'IVCAD Geral':                                    'IVCAD',
        'IVCAD — Disponibilidade de Recursos':            'IVCAD_DR',
        'IVCAD — Trabalho e Qualificação de Adultos':     'IVCAD_TQA',
        'IVCAD — Necessidade de Cuidados':                'IVCAD_NC',
        'IVCAD — Condições Habitacionais':                'IVCAD_CH',
        'IVCAD — Desenvolvimento Crianças/Adolescentes':  'IVCAD_DCA',
        'IVCAD — Desenvolvimento Primeira Infância':      'IVCAD_DPI',
    },
    'EDUCAÇÃO (Censo 2010)': {
        'Expectativa de Anos de Estudo (18 anos)':      'E_ANOSESTUDO',
        '% com Ensino Fundamental Completo (25+ anos)': 'PERC_FUND_COMP',
        '% com Ensino Médio Completo (25+ anos)':       'PERC_MED_COMP',
    },
    'QUALIDADE DO ENSINO MÉDIO PÚBLICO (Censo Escolar 2023)': {
        'Nota Média SAEB — Ens. Médio Público':          'NOTA_MEDIA_SAEB_2023',
        'IDEB — Ensino Médio Público':                   'IDEB_2023',
        'Indicador de Rendimento — Ens. Médio Público':  'IND_REND_2023',
    },
    'DESENVOLVIMENTO HUMANO (Censo 2010)': {
        'IDHM Geral':    'IDHM',
        'IDHM Renda':    'IDHM_R',
        'IDHM Educação': 'IDHM_E',
    },
    'GASTOS EDUCACIONAIS (SIOPE 2023)': {
        '% de Impostos Aplicados em Educação':  'EDU_Perc_Aplicacao',
        'Investimento por Aluno (R$)':          'EDU_Investimento_Aluno',
        'Total Aplicado em Educação (R$)':      'EDU_Aplicacao_Total',
        'Total de Alunos Matriculados':         'EDU_Alunos',
        'Receita FUNDEB (R$)':                  'EDU_ReceitaFUNDEB',
    },
}

available_metrics = {}
item_categories   = {}
for cat, items in INDICATORS_DICT.items():
    for label, col in items.items():
        if col in df.columns and df[col].count() > 1:
            dlabel = f"[{cat}] {label}"
            available_metrics[dlabel] = col
            item_categories[dlabel]   = cat

# ── HELPERS ───────────────────────────────────────────────────────
def safe_scatter_df(df, x_col, y_col, extra_cols=None):
    base = ['Nome_Municipio', 'QTD_CANDIDATOS']
    if extra_cols:
        base += [c for c in extra_cols if c not in [x_col, y_col] and c in df.columns]
    all_cols = list(dict.fromkeys([x_col, y_col] + base))
    return df[all_cols].dropna().reset_index(drop=True)

def corr_label(r, p):
    """Retorna emoji + descrição para o dropdown."""
    if abs(r) < 0.3:    return "↔", "Fraca"
    if r > 0:           return "🔺", "Positiva"
    return "🔻", "Negativa"

def sig_icon(p):
    return "✅" if p < 0.05 else "❌"

# ── CARDS DE OVERVIEW ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-icon"><span class="material-symbols-outlined">map</span></div><div class="metric-title">Municípios Analisados</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-icon"><span class="material-symbols-outlined">person</span></div><div class="metric-title">Total de Candidatos</div><div class="metric-value">{int(df["QTD_CANDIDATOS"].sum()):,}</div></div>', unsafe_allow_html=True)
with c3:
    idx = df["NOTA_MEDIA"].idxmax()
    st.markdown(f'<div class="metric-card"><div class="metric-icon"><span class="material-symbols-outlined">emoji_events</span></div><div class="metric-title">Maior Nota Média</div><div class="metric-value">{df.loc[idx,"NOTA_MEDIA"]:.1f}</div><div class="metric-sub">{df.loc[idx,"Nome_Municipio"]}</div></div>', unsafe_allow_html=True)
with c4:
    idx = df["NOTA_MEDIA"].idxmin()
    st.markdown(f'<div class="metric-card"><div class="metric-icon"><span class="material-symbols-outlined">trending_down</span></div><div class="metric-title">Menor Nota Média</div><div class="metric-value">{df.loc[idx,"NOTA_MEDIA"]:.1f}</div><div class="metric-sub">{df.loc[idx,"Nome_Municipio"]}</div></div>', unsafe_allow_html=True)

# ── DADOS TEMPORAIS (carregados uma vez, sem depender do filtro de escola) ──
# ALTERAÇÃO: leitura segura do CSV temporal usando o mesmo tratamento de erros
@st.cache_data
def load_temporal():
    dados_temporais, erro_temporal = load_csv_data("dados_temporal.csv")
    return dados_temporais, erro_temporal


df_temporal, erro_temporal = load_temporal()

# ── ABAS ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    ":material/scatter_plot: Explorador de Correlação",
    ":material/science: Testes Estatísticos",
    ":material/manage_search: Outliers e Distribuição",
    ":material/smart_toy: Machine Learning",
    ":material/timeline: Evolução Temporal",
])

# ─── TAB 1 ────────────────────────────────────────────────────────
with tab1:
    st.markdown("### :material/analytics: Explorador de Correlação")
    st.markdown("Selecione dois indicadores para visualizar a relação entre eles. O tamanho dos pontos representa o número de candidatos.")

    cx, cy = st.columns(2)
    with cx:
        st.markdown("<p style='color:#94A3B8;margin-bottom:0'>Eixo X (variável independente):</p>", unsafe_allow_html=True)
        cat_x_idx = list(INDICATORS_DICT.keys()).index(st.session_state['cat_x']) \
                    if st.session_state['cat_x'] in INDICATORS_DICT else 0
        cat_x   = st.selectbox("Cat X:", list(INDICATORS_DICT.keys()),
                                index=cat_x_idx, key='cat_x', label_visibility="collapsed")
        opts_x  = {k: v for k, v in INDICATORS_DICT[cat_x].items()
                   if v in df.columns and df[v].count() > 1}
        # Default: Nota Média Geral se X for NOTAS DO ENEM
        default_x_idx = list(opts_x.keys()).index('Nota Média Geral') \
                        if 'Nota Média Geral' in opts_x else 0
        label_x = st.selectbox("Ind X:", list(opts_x.keys()),
                                index=default_x_idx, key='ind_x', label_visibility="collapsed")
        x_col   = opts_x[label_x]

    with cy:
        st.markdown("<p style='color:#94A3B8;margin-bottom:0'>Eixo Y (variável dependente):</p>", unsafe_allow_html=True)
        cats_y        = [c for c in INDICATORS_DICT.keys() if c != cat_x]
        cat_y_default = st.session_state['cat_y'] if st.session_state['cat_y'] in cats_y else cats_y[0]
        default_idx   = cats_y.index(cat_y_default)
        cat_y         = st.selectbox("Cat Y:", cats_y, index=default_idx,
                                     key='cat_y', label_visibility="collapsed")

        opts_y = {}
        for lbl, col in INDICATORS_DICT[cat_y].items():
            if col not in df.columns or df[col].count() <= 1: continue
            r = df[x_col].corr(df[col])
            if pd.isna(r): continue
            arrow, strength = corr_label(r, 1)
            opts_y[f"{arrow} {lbl} ({strength})"] = col

        if not opts_y:
            st.warning("Sem dados correlacionáveis nesta categoria.")
            st.stop()

        # Default: IDHM Geral se Y for DESENVOLVIMENTO HUMANO
        opts_y_labels  = list(opts_y.keys())
        default_y_lbl  = next((l for l in opts_y_labels if 'IDHM Geral' in l), opts_y_labels[0])
        default_y_idx  = opts_y_labels.index(default_y_lbl)
        label_y_full   = st.selectbox("Ind Y:", opts_y_labels,
                                      index=default_y_idx, key='ind_y', label_visibility="collapsed")
        y_col          = opts_y[label_y_full]
        y_axis_label   = label_y_full.split('(')[0][2:].strip()

    # Nota sobre % aplicação em educação
    if x_col == 'EDU_Perc_Aplicacao' or y_col == 'EDU_Perc_Aplicacao':
        if 'EDU_Perc_Aplicacao' in df.columns:
            abaixo = (df['EDU_Perc_Aplicacao'] < 25).sum()
            st.markdown(
                f'<div class="info-banner"><span class="material-symbols-outlined" style="vertical-align: middle; font-size: 1.1em;">info</span> <b>% de Impostos Aplicados em Educação:</b> '
                f'A Constituição Federal exige mínimo de <b>25%</b> para municípios. '
                f'No ES 2023: mín {df["EDU_Perc_Aplicacao"].min():.1f}% | '
                f'máx {df["EDU_Perc_Aplicacao"].max():.1f}% | '
                f'<b>{abaixo} município(s) abaixo do mínimo constitucional.</b></div>',
                unsafe_allow_html=True
            )

    sub          = df[[x_col, y_col]].dropna()
    r_val, p_val = stats.pearsonr(sub[x_col], sub[y_col])
    r2           = r_val ** 2

    if r_val > 0.6:    corr_desc = '🟢 Forte e Positiva'
    elif r_val < -0.6: corr_desc = '🔴 Forte e Negativa'
    elif abs(r_val) > 0.3: corr_desc = '🟡 Moderada'
    else:              corr_desc = '⚪ Fraca / Inexistente'

    k1, k2, k3 = st.columns(3)
    k1.markdown(f'<div class="stat-box">📐 <b>Pearson r</b><br><span style="font-size:1.4rem;color:#38BDF8">{r_val:+.3f}</span><br>{corr_desc}</div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="stat-box">📊 <b>R² (variância explicada)</b><br><span style="font-size:1.4rem;color:#38BDF8">{r2:.1%}</span></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="stat-box">🔬 <b>p-valor (α = 0,05)</b><br><span style="font-size:1.4rem;color:#38BDF8">{p_val:.4f}</span><br>{"✓ Significativo" if p_val < 0.05 else "✗ Não significativo"}</div>', unsafe_allow_html=True)

    if abs(r_val) < 0.3:
        st.caption(
            f"A relação entre {label_x} e {y_axis_label} é fraca nos dados atuais (r = {r_val:+.3f}), "
            f"o que sugere que esse indicador, por si só, não explica variações de desempenho entre os "
            f"{len(sub)} municípios analisados. Isso pode refletir a homogeneidade das condições entre "
            f"municípios nesse recorte, ou que outros fatores têm peso maior. "
            f"Experimente cruzar com IDHM ou vulnerabilidade social para padrões mais claros."
        )

    use_idhm = 'IDHM' in df.columns and x_col != 'IDHM' and y_col != 'IDHM'
    hover_df = safe_scatter_df(df, x_col, y_col, ['IDHM'] if use_idhm else [])

    fig = px.scatter(
        hover_df, x=x_col, y=y_col,
        hover_name="Nome_Municipio", size="QTD_CANDIDATOS",
        color="IDHM" if use_idhm else None,
        color_continuous_scale=px.colors.sequential.Tealgrn,
        trendline="ols",
        labels={x_col: label_x, y_col: y_axis_label},
        title=f"{y_axis_label} × {label_x}  |  r = {r_val:+.3f}  |  R² = {r2:.1%}",
        template="plotly_dark",
    )
    for ax, col in [('y', y_col), ('x', x_col)]:
        if col == 'EDU_Perc_Aplicacao':
            if ax == 'y':
                fig.add_hline(y=25, line_dash="dash", line_color="#F97316",
                              annotation_text="Mínimo constitucional (25%)", annotation_position="top right")
            else:
                fig.add_vline(x=25, line_dash="dash", line_color="#F97316",
                              annotation_text="Mínimo constitucional (25%)", annotation_position="top right")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#E2E8F0"), height=520)

    # Traduzir o hover da linha de tendência OLS para português
    for trace in fig.data:
        if hasattr(trace, 'name') and trace.name == 'OLS trendline':
            trace.name = 'Linha de tendência (OLS)'
            trace.hovertemplate = (
                f"<b>Linha de tendência</b><br>"
                f"{label_x}: %{{x:.3f}}<br>"
                f"{y_axis_label} estimado: %{{y:.2f}}<br>"
                f"<i>r = {r_val:+.3f} | R² = {r2:.1%}</i>"
                "<extra></extra>"
            )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### :material/folder_open: Dados por Município")
    disp = [c for c in ['Nome_Municipio', 'QTD_CANDIDATOS', 'PERC_ESCOLA_PUB', 'NOTA_MEDIA',
                         'IDHM', 'IDHM_R', 'IDHM_E', 'GINI', 'TX_ANALF', 'NEET_VULN',
                         'IVCAD', 'IVCAD_DR', 'IVCAD_TQA',
                         'EDU_Perc_Aplicacao', 'EDU_Investimento_Aluno', 'EDU_Aplicacao_Total']
            if c in df.columns]
    st.dataframe(df[disp].sort_values('NOTA_MEDIA', ascending=False).reset_index(drop=True),
                 use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### :material/grid_view: Matriz de Correlação entre Indicadores")
    st.markdown("Valores próximos de **+1** = correlação positiva forte; **-1** = negativa forte; **0** = sem relação.")

    grupos = st.multiselect("Grupos a incluir:", list(INDICATORS_DICT.keys()),
                            default=list(INDICATORS_DICT.keys()), key='grupos_matriz')
    cols_m, labels_m = [], {}
    for cat in grupos:
        for lbl, col in INDICATORS_DICT[cat].items():
            if col in df.columns and df[col].count() > 1 and col not in cols_m:
                cols_m.append(col)
                labels_m[col] = lbl

    if len(cols_m) < 2:
        st.warning("Selecione ao menos dois grupos.")
    else:
        corr_m = df[cols_m].corr()
        corr_m.index   = [labels_m.get(c, c) for c in corr_m.index]
        corr_m.columns = [labels_m.get(c, c) for c in corr_m.columns]
        fig_m = px.imshow(corr_m, text_auto=".2f", aspect="auto",
                          color_continuous_scale="RdBu", zmin=-1, zmax=1,
                          origin="lower", template="plotly_dark")
        fig_m.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            height=700, font=dict(color="#E2E8F0", size=11))
        st.plotly_chart(fig_m, use_container_width=True)

# ─── TAB 2 ────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🧪 Testes Estatísticos")

    # Regressão Linear
    st.markdown("#### :material/show_chart: Regressão Linear Simples")
    r_c1, r_c2 = st.columns(2)
    with r_c1:
        reg_cats_x = [c for c in INDICATORS_DICT if c != 'NOTAS DO ENEM']
        reg_cat_x  = st.selectbox("Variável independente (X):", reg_cats_x, key='reg_cat_x')

        # FIX 3: adiciona ícone de significância nos indicadores do dropdown
        reg_opts_x_raw = {k: v for k, v in INDICATORS_DICT[reg_cat_x].items()
                          if v in df.columns and df[v].count() > 1}
        reg_opts_x = {}
        for lbl, col in reg_opts_x_raw.items():
            sub_r = df[['NOTA_MEDIA', col]].dropna()
            if len(sub_r) >= 5:
                _, p_r = stats.pearsonr(sub_r['NOTA_MEDIA'], sub_r[col])
                icon = sig_icon(p_r)
            else:
                icon = "❓"
            reg_opts_x[f"{icon} {lbl}"] = col

        reg_lbl_x = st.selectbox("Indicador X:", list(reg_opts_x.keys()), key='reg_ind_x')
        reg_x     = reg_opts_x[reg_lbl_x]

    with r_c2:
        reg_opts_y = {k: v for k, v in INDICATORS_DICT['NOTAS DO ENEM'].items()
                      if v in df.columns and df[v].count() > 1}
        reg_lbl_y  = st.selectbox("Variável dependente (Y — nota):", list(reg_opts_y.keys()), key='reg_ind_y')
        reg_y      = reg_opts_y[reg_lbl_y]

    if reg_x == 'EDU_Perc_Aplicacao':
        st.markdown('<div class="info-banner">ℹ️ O mínimo constitucional de aplicação em educação é <b>25%</b> para municípios.</div>', unsafe_allow_html=True)

    reg_sub = df[[reg_x, reg_y, 'Nome_Municipio', 'QTD_CANDIDATOS']].dropna().reset_index(drop=True)
    slope, intercept, r_reg, p_reg, se = stats.linregress(reg_sub[reg_x], reg_sub[reg_y])
    r2_reg = r_reg ** 2

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R²", f"{r2_reg:.1%}")
    m2.metric("r de Pearson", f"{r_reg:+.3f}")
    m3.metric("p-valor", f"{p_reg:.4f}")
    m4.metric("Coef. angular", f"{slope:.4f}")

    st.markdown(f"""
    **Interpretação:** *{reg_lbl_x.split(' ', 1)[-1]}* explica **{r2_reg:.1%}** da variação em *{reg_lbl_y}* entre os {len(reg_sub)} municípios.
    {'Resultado estatisticamente significativo (p < 0,05).' if p_reg < 0.05 else 'Resultado **não** estatisticamente significativo (p ≥ 0,05).'}
    A cada unidade de aumento em *{reg_lbl_x.split(' ', 1)[-1]}*, a nota varia em média **{slope:+.4f}** pontos.
    """)

    reg_hover = safe_scatter_df(df, reg_x, reg_y)
    fig_reg = px.scatter(reg_hover, x=reg_x, y=reg_y, hover_name="Nome_Municipio",
                         size="QTD_CANDIDATOS", trendline="ols",
                         labels={reg_x: reg_lbl_x.split(' ', 1)[-1], reg_y: reg_lbl_y},
                         title=f"Regressão: {reg_lbl_y} ~ {reg_lbl_x.split(' ', 1)[-1]}  |  R² = {r2_reg:.1%}  |  p = {p_reg:.4f}",
                         template="plotly_dark")
    if reg_x == 'EDU_Perc_Aplicacao':
        fig_reg.add_vline(x=25, line_dash="dash", line_color="#F97316",
                          annotation_text="Mínimo constitucional (25%)", annotation_position="top right")
    fig_reg.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#E2E8F0"), height=450)
    st.plotly_chart(fig_reg, use_container_width=True)

    st.markdown("---")

    # Teste t
    st.markdown("#### :material/biotech: Teste t — Comparação de Grupos")
    st.markdown("Compara a nota média entre municípios com **alto** e **baixo** valor de um indicador (divisão pela mediana).")

    t_c1, t_c2 = st.columns(2)
    with t_c1:
        t_cats = [c for c in INDICATORS_DICT if c != 'NOTAS DO ENEM']
        t_cat  = st.selectbox("Categoria do indicador:", t_cats, key='t_cat')
        t_opts = {k: v for k, v in INDICATORS_DICT[t_cat].items()
                  if v in df.columns and df[v].count() > 1}
        t_lbl  = st.selectbox("Indicador:", list(t_opts.keys()), key='t_ind')
        t_col  = t_opts[t_lbl]
    with t_c2:
        n_opts = {k: v for k, v in INDICATORS_DICT['NOTAS DO ENEM'].items()
                  if v in df.columns and df[v].count() > 1}
        n_lbl  = st.selectbox("Nota a comparar:", list(n_opts.keys()), key='t_nota')
        n_col  = n_opts[n_lbl]

    t_sub    = df[[t_col, n_col, 'Nome_Municipio']].dropna().reset_index(drop=True)
    mediana  = t_sub[t_col].median()
    g_alto   = t_sub[t_sub[t_col] >= mediana][n_col]
    g_baixo  = t_sub[t_sub[t_col] <  mediana][n_col]
    t_stat, t_p = stats.ttest_ind(g_alto, g_baixo)

    ta, tb, tc, td = st.columns(4)
    ta.metric("Média — Alto",  f"{g_alto.mean():.2f}",  delta=f"n={len(g_alto)}")
    tb.metric("Média — Baixo", f"{g_baixo.mean():.2f}", delta=f"n={len(g_baixo)}")
    tc.metric("Estatística t", f"{t_stat:.3f}")
    td.metric("p-valor",       f"{t_p:.4f}")

    st.markdown(f"""
    **Interpretação:** Municípios com **alto** *{t_lbl}* têm nota média de **{g_alto.mean():.2f}**,
    contra **{g_baixo.mean():.2f}** com **baixo** *{t_lbl}* (diferença: {abs(g_alto.mean()-g_baixo.mean()):.2f} pontos).
    {'Diferença **estatisticamente significativa** (p < 0,05).' if t_p < 0.05 else 'Diferença **não** estatisticamente significativa (p ≥ 0,05).'}
    """)

    t_sub2 = t_sub.assign(Grupo=t_sub[t_col].apply(
        lambda v: f"Alto {t_lbl[:15]}" if v >= mediana else f"Baixo {t_lbl[:15]}"))
    fig_t = px.box(t_sub2, x="Grupo", y=n_col, points="all", hover_name="Nome_Municipio",
                   color="Grupo",
                   title=f"{n_lbl} por grupo de {t_lbl}  |  t = {t_stat:.3f}  |  p = {t_p:.4f}",
                   template="plotly_dark", color_discrete_sequence=["#38BDF8", "#F97316"])
    fig_t.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E2E8F0"), height=420, showlegend=False)
    st.plotly_chart(fig_t, use_container_width=True)

    st.markdown("---")

    # Tabela completa de correlações
    st.markdown("#### :material/table: Tabela de Correlações com a Nota Média Geral")
    rows = []
    for cat, items in INDICATORS_DICT.items():
        if cat == 'NOTAS DO ENEM': continue
        for lbl, col in items.items():
            sub2 = df[['NOTA_MEDIA', col]].dropna()
            if len(sub2) < 5: continue
            r2c, p2c = stats.pearsonr(sub2['NOTA_MEDIA'], sub2[col])
            rows.append({'Grupo': cat, 'Indicador': lbl,
                         'r (Pearson)': round(r2c, 3), 'R²': f"{r2c**2:.1%}",
                         'p-valor': round(p2c, 4),
                         'Significativo': '✓' if p2c < 0.05 else '✗',
                         'n': len(sub2)})
    df_ct = pd.DataFrame(rows).sort_values('r (Pearson)', key=abs, ascending=False)
    st.dataframe(df_ct, use_container_width=True, hide_index=True)

# ─── TAB 3 ────────────────────────────────────────────────────────
with tab3:
    st.markdown("### :material/track_changes: Detecção de Outliers e Distribuição")

    # FIX 4: opção de método de detecção de outliers
    met_col, out_col_sel = st.columns([1, 3])
    with met_col:
        metodo = st.radio("Método:", ["IQR", "Z-score (>2σ)"], key="out_metodo",
                          help="IQR: Q3+1.5×IQR. Z-score: valores a mais de 2 desvios padrões da média.")
    with out_col_sel:
        out_lbl     = st.selectbox("Indicador:", list(available_metrics.keys()), key='outlier1')

    out_col     = available_metrics[out_lbl]
    out_display = out_lbl.split('] ')[-1]

    # FIX 6: garantir colunas únicas no df_out
    aux_cols = [c for c in ['Nome_Municipio', 'NOTA_MEDIA', 'IDHM', 'QTD_CANDIDATOS']
                if c != out_col and c in df.columns]
    df_out = df[[out_col] + aux_cols].dropna(subset=[out_col]).reset_index(drop=True)

    cb, cbar = st.columns(2)
    with cb:
        fig_box = px.box(df_out, y=out_col, points="all", hover_name="Nome_Municipio",
                         title=f"Boxplot — {out_display}", template="plotly_dark")
        fig_box.update_traces(marker=dict(size=7, color="#38BDF8"), boxmean=True)
        if out_col == 'EDU_Perc_Aplicacao':
            fig_box.add_hline(y=25, line_dash="dash", line_color="#F97316",
                              annotation_text="Mínimo 25%", annotation_position="top right")
        fig_box.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#E2E8F0"))
        st.plotly_chart(fig_box, use_container_width=True)

    with cbar:
        idhm_col    = 'IDHM' if 'IDHM' in df_out.columns and out_col != 'IDHM' else None
        df_sorted   = df_out.sort_values(out_col, ascending=False)
        df_extremes = pd.concat([df_sorted.head(10), df_sorted.tail(5)]).reset_index(drop=True)
        fig_bar = px.bar(df_extremes, x="Nome_Municipio", y=out_col,
                         color=idhm_col,
                         title=f"Top 10 maiores / 5 menores — {out_display}",
                         color_continuous_scale="Viridis", template="plotly_dark",
                         category_orders={"Nome_Municipio": df_extremes["Nome_Municipio"].tolist()})
        if out_col == 'EDU_Perc_Aplicacao':
            fig_bar.add_hline(y=25, line_dash="dash", line_color="#F97316",
                              annotation_text="Mínimo 25%", annotation_position="top right")
        fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#E2E8F0"), xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### Estatísticas Descritivas")
    desc = df_out[out_col].describe()
    sd1, sd2, sd3, sd4, sd5 = st.columns(5)
    sd1.metric("Média",         f"{desc['mean']:.2f}")
    sd2.metric("Mediana",       f"{df_out[out_col].median():.2f}")
    sd3.metric("Desvio padrão", f"{desc['std']:.2f}")
    sd4.metric("Mínimo",        f"{desc['min']:.2f}")
    sd5.metric("Máximo",        f"{desc['max']:.2f}")

    # FIX 4: dois métodos de detecção
    st.markdown(f"#### Municípios Identificados como Outliers (método {metodo})")

    if metodo == "IQR":
        q1, q3  = df_out[out_col].quantile(0.25), df_out[out_col].quantile(0.75)
        iqr     = q3 - q1
        lim_inf = q1 - 1.5 * iqr
        lim_sup = q3 + 1.5 * iqr
        outliers = df_out[(df_out[out_col] < lim_inf) | (df_out[out_col] > lim_sup)].reset_index(drop=True)
        st.caption(f"Limites IQR: [{lim_inf:.2f}, {lim_sup:.2f}]  |  Para {out_display}, um valor de {desc['max']:.2f} precisaria ultrapassar {lim_sup:.2f} para ser outlier pelo IQR.")
    else:
        mean_v = df_out[out_col].mean()
        std_v  = df_out[out_col].std()
        outliers = df_out[np.abs(df_out[out_col] - mean_v) > 2 * std_v].reset_index(drop=True)
        st.caption(f"Limites Z-score: [{mean_v - 2*std_v:.2f}, {mean_v + 2*std_v:.2f}]  (média ± 2σ)")

    if outliers.empty:
        st.info(f"Nenhum outlier detectado pelo método {metodo} para este indicador.")
    else:
        # FIX 6: reset_index antes do sort_values para evitar ValueError
        show = list(dict.fromkeys([c for c in ['Nome_Municipio', out_col, 'NOTA_MEDIA', 'IDHM', 'QTD_CANDIDATOS']
                                   if c in outliers.columns]))
        st.dataframe(
            outliers[show].reset_index(drop=True).sort_values(out_col, ascending=False).reset_index(drop=True),
            use_container_width=True, hide_index=True
        )

# ─── TAB 4: MACHINE LEARNING ─────────────────────────────────────
with tab4:
    st.markdown("### :material/smart_toy: Machine Learning — Agrupamento e Importância de Variáveis")
    st.markdown(
        "Aplicação de dois algoritmos de Machine Learning: "
        "**K-Means** para agrupar municípios em perfis socioeconômicos semelhantes "
        "e **Random Forest** para identificar quais variáveis mais explicam o desempenho no ENEM."
    )

    # Selecionar features disponíveis
    ml_features = [c for c in [
        'IDHM', 'IDHM_R', 'IDHM_E', 'GINI', 'TX_ANALF', 'NEET_VULN', 'CRIAN_VULN',
        'IVCAD', 'IVCAD_DR', 'IVCAD_TQA',
        'NOTA_MEDIA_SAEB_2023', 'IDEB_2023',
        'EDU_Investimento_Aluno', 'EDU_Perc_Aplicacao',
    ] if c in df.columns and df[c].count() > 10]

    df_ml = df[ml_features + ['NOTA_MEDIA', 'Nome_Municipio', 'QTD_CANDIDATOS']].dropna().reset_index(drop=True)

    if len(df_ml) < 20:
        st.warning(f"Amostra insuficiente para Machine Learning ({len(df_ml)} municípios). Reduza o filtro de candidatos mínimos.")
        st.stop()

    st.caption(f"Modelos treinados com **{len(df_ml)} municípios** e **{len(ml_features)} variáveis** socioeconômicas.")

    # ── SEÇÃO 1: K-MEANS ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### :material/track_changes: K-Means — Agrupamento de Municípios em Perfis")
    st.markdown(
        "O algoritmo K-Means agrupa municípios com características socioeconômicas e de desempenho "
        "semelhantes em **clusters**. Cada cluster representa um perfil distinto de município no ES."
    )

    n_clusters = st.slider("Número de clusters:", min_value=2, max_value=6, value=4, key='n_clusters',
                            help="3-4 clusters geralmente é o ideal para 78 municípios.")

    # Padronizar e clusterizar
    X_clust = df_ml[ml_features + ['NOTA_MEDIA']].values
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_clust)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_ml['Cluster'] = km.fit_predict(X_sc)

    # Renomear clusters por ordem de nota média (Cluster 0 = pior, Cluster N = melhor)
    cluster_order = df_ml.groupby('Cluster')['NOTA_MEDIA'].mean().sort_values().index.tolist()
    rename_map    = {old: new for new, old in enumerate(cluster_order)}
    df_ml['Cluster'] = df_ml['Cluster'].map(rename_map)

    # PCA para visualização 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sc)
    df_ml['PCA1'] = X_pca[:, 0]
    df_ml['PCA2'] = X_pca[:, 1]

    # Perfil dos clusters
    perfil = df_ml.groupby('Cluster').agg(
        Municípios=('Nome_Municipio', 'count'),
        Nota_Média=('NOTA_MEDIA', 'mean'),
        IDHM_Médio=('IDHM', 'mean') if 'IDHM' in df_ml.columns else ('NOTA_MEDIA', 'count'),
    ).round(3).reset_index()
    perfil.columns = ['Cluster', 'Nº Municípios', 'Nota Média ENEM', 'IDHM Médio']

    pc1, pc2 = st.columns([1, 2])
    with pc1:
        st.markdown("**Perfil de cada cluster**")
        st.dataframe(perfil, use_container_width=True, hide_index=True)
        st.caption("Clusters ordenados do menor (0) para o maior (N) em nota média.")

    with pc2:
        # Visualização PCA com clusters
        df_ml['Cluster_str'] = "Cluster " + df_ml['Cluster'].astype(str)
        fig_clust = px.scatter(
            df_ml, x='PCA1', y='PCA2',
            color='Cluster_str',
            size='QTD_CANDIDATOS',
            hover_name='Nome_Municipio',
            hover_data={'NOTA_MEDIA': ':.1f', 'IDHM': ':.3f', 'PCA1': False, 'PCA2': False, 'Cluster_str': False},
            title="Agrupamento dos municípios (visualização via PCA)",
            labels={'PCA1': 'Componente Principal 1', 'PCA2': 'Componente Principal 2'},
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_clust.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"), height=450,
            legend=dict(title="", orientation="h", yanchor="bottom", y=-0.2),
        )
        st.plotly_chart(fig_clust, use_container_width=True)
        st.caption(f"PCA explica {pca.explained_variance_ratio_.sum():.1%} da variação total dos dados em 2 dimensões.")

    # Lista de municípios por cluster
    st.markdown("**Municípios por cluster:**")
    for c in sorted(df_ml['Cluster'].unique()):
        muns      = df_ml[df_ml['Cluster'] == c].sort_values('NOTA_MEDIA', ascending=False)
        nota_avg  = muns['NOTA_MEDIA'].mean()
        idhm_avg  = muns['IDHM'].mean() if 'IDHM' in muns.columns else None
        cor       = px.colors.qualitative.Bold[c % len(px.colors.qualitative.Bold)]
        idhm_txt  = f" | IDHM médio {idhm_avg:.3f}" if idhm_avg else ""
        with st.expander(f"Cluster {c} — {len(muns)} municípios | Nota média {nota_avg:.1f}{idhm_txt}"):
            lista = ", ".join(muns['Nome_Municipio'].tolist())
            st.markdown(f"<span style='color:{cor}'>●</span> {lista}", unsafe_allow_html=True)

    # ── SEÇÃO 2: RANDOM FOREST ───────────────────────────────────
    st.markdown("---")
    st.markdown("#### :material/forest: Random Forest — Importância das Variáveis")
    st.markdown(
        "O Random Forest é um modelo de ensemble que ranqueia quais variáveis "
        "mais influenciam a nota do ENEM. Diferente da correlação simples, ele "
        "captura **relações não-lineares** e interações entre variáveis."
    )

    y_rf = df_ml['NOTA_MEDIA']
    X_rf = df_ml[ml_features]

    rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=8)
    rf.fit(X_rf, y_rf)
    r2_rf = rf.score(X_rf, y_rf)

    # Importância das variáveis
    importancia = pd.DataFrame({
        'Variável': X_rf.columns,
        'Importância (%)': rf.feature_importances_ * 100,
    }).sort_values('Importância (%)', ascending=True)

    # Renomear variáveis para nomes amigáveis
    nome_amigavel = {}
    for cat, items in INDICATORS_DICT.items():
        for lbl, col in items.items():
            nome_amigavel[col] = lbl
    importancia['Variável'] = importancia['Variável'].map(lambda v: nome_amigavel.get(v, v))

    rfc1, rfc2 = st.columns([1, 2])
    with rfc1:
        st.markdown("**Métricas do modelo**")
        st.metric("R² (variância explicada)", f"{r2_rf:.1%}")
        st.metric("Variáveis utilizadas",     f"{len(ml_features)}")
        st.metric("Municípios no modelo",     f"{len(df_ml)}")
        st.caption(
            "O R² alto reflete bom ajuste aos dados, "
            "mas com 78 municípios o modelo é melhor "
            "para identificar **importância de variáveis** "
            "do que para previsões de novos casos."
        )

    with rfc2:
        fig_imp = px.bar(
            importancia, x='Importância (%)', y='Variável',
            orientation='h',
            title="Variáveis mais importantes para explicar a nota do ENEM",
            template="plotly_dark",
            color='Importância (%)',
            color_continuous_scale='Tealgrn',
        )
        fig_imp.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"), height=500,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # Interpretação automática
    top3 = importancia.tail(3).iloc[::-1]['Variável'].tolist()
    st.markdown(
        f"""
        **Interpretação:** as três variáveis mais determinantes para o desempenho no ENEM
        nos municípios do ES, segundo o Random Forest, são:
        **{top3[0]}**, **{top3[1]}** e **{top3[2]}**.
        Isso confirma e quantifica os achados das análises de correlação:
        a qualidade do ensino e o contexto socioeconômico têm maior peso do que
        o gasto público em educação isoladamente.
        """
    )

    # ── SEÇÃO 3: Previsão 2025 ────────────────────────────────────
    st.markdown("---")
    st.markdown("#### :material/online_prediction: Previsão para o ENEM 2025")
    st.markdown(
        "Regressão linear temporal aplicada à série histórica 2016–2024 de cada município. "
        "A tendência é extrapolada para 2025 com intervalo de confiança de 95%."
    )
    st.markdown(
        '<div class="warn-banner">⚠️ <b>Limitação metodológica:</b> previsão baseada em tendência linear com ~9 pontos por município. '
        'Fatores externos (mudanças no ENEM, políticas educacionais, eventos imprevistos) não são capturados. '
        'Use como estimativa de tendência, não como previsão precisa.</div>',
        unsafe_allow_html=True
    )

    if df_temporal is None:
        st.info(erro_temporal or "Execute `python process_temporal.py` para habilitar a previsão 2025.")
    else:
        from scipy.stats import t as t_dist
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        df_t_ml = df_temporal.copy()
        df_t_ml['Ano'] = df_t_ml['Ano'].astype(int)
        anos_disp_ml = sorted(df_t_ml['Ano'].unique())

        # Indicadores socioeconômicos disponíveis para enriquecer o modelo
        FEAT_SOCIO = {
            'Investimento por Aluno (SIOPE)': 'EDU_Investimento_Aluno',
            'IDHM Geral':                     'IDHM',
            'Nota SAEB — Ens. Médio':         'NOTA_MEDIA_SAEB_2023',
            'IVCAD — Vulnerabilidade':        'IVCAD',
            'Taxa de Analfabetismo':          'TX_ANALF',
        }
        feats_disponiveis = {k: v for k, v in FEAT_SOCIO.items()
                             if df_raw is not None and v in df_raw.columns
                             and df_raw[v].count() > 10}

        # Controles — coluna da direita
        prev_col1, prev_col2 = st.columns([2, 1])
        with prev_col2:
            anos_base = st.multiselect(
                "Anos usados no modelo:",
                options=anos_disp_ml,
                default=[a for a in anos_disp_ml if a >= 2016],
                key='prev_anos',
                help="Recomendado: a partir de 2016, quando a escala TRI ficou mais estável.",
            )
            prev_nota = st.selectbox(
                "Nota a prever:",
                options=['Nota Média Geral', 'Matemática', 'Redação',
                         'Linguagens', 'Ciências Humanas', 'Ciências da Natureza'],
                key='prev_nota_sel',
            )
            prev_nota_col = {
                'Nota Média Geral': 'NOTA_MEDIA', 'Matemática': 'NU_NOTA_MT',
                'Redação': 'NU_NOTA_REDACAO', 'Linguagens': 'NU_NOTA_LC',
                'Ciências Humanas': 'NU_NOTA_CH', 'Ciências da Natureza': 'NU_NOTA_CN',
            }[prev_nota]

            # Features extras do modelo
            if feats_disponiveis:
                feats_sel_labels = st.multiselect(
                    "Indicadores adicionais no modelo:",
                    options=list(feats_disponiveis.keys()),
                    default=list(feats_disponiveis.keys())[:3],
                    key='prev_feats',
                    help="Combinados com o ano para gerar uma previsão multivariada (regressão de painel).",
                )
                feats_sel_cols = [feats_disponiveis[k] for k in feats_sel_labels]
            else:
                feats_sel_cols = []

        if len(anos_base) < 3:
            st.warning("Selecione ao menos 3 anos para ajustar o modelo.")
        else:
            ANO_PREV = 2025
            df_base = df_t_ml[df_t_ml['Ano'].isin(anos_base)].copy()

            # ── 1. Regressão temporal por município (IC 95%) ───────
            previsoes = []
            for mun, grp in df_base.groupby('NO_MUNICIPIO_PROVA'):
                serie = grp[['Ano', prev_nota_col]].dropna().sort_values('Ano')
                if len(serie) < 3:
                    continue
                x = serie['Ano'].values
                y = serie[prev_nota_col].values
                slope, intercept, r, p, se = stats.linregress(x, y)
                n, x_mean = len(x), x.mean()
                ss_x    = ((x - x_mean) ** 2).sum()
                se_pred = se * np.sqrt(1 + 1/n + (ANO_PREV - x_mean)**2 / ss_x)
                t_crit  = t_dist.ppf(0.975, df=n - 2)
                y_pred  = slope * ANO_PREV + intercept
                previsoes.append({
                    'NO_MUNICIPIO_PROVA': mun,
                    'Município': mun,
                    f'Nota {anos_base[-1]}': round(
                        grp[grp['Ano'] == anos_base[-1]][prev_nota_col].values[0]
                        if anos_base[-1] in grp['Ano'].values else np.nan, 2),
                    'Previsão 2025 (tendência)': round(y_pred, 2),
                    'IC inferior (95%)': round(y_pred - t_crit * se_pred, 2),
                    'IC superior (95%)': round(y_pred + t_crit * se_pred, 2),
                    'Tendência anual': round(slope, 2),
                    'R² temporal': round(r**2, 3),
                    '_slope': slope, '_intercept': intercept,
                })
            df_prev = pd.DataFrame(previsoes)

            # ── 2. Modelo de painel (Ano + indicadores socio) ──────
            usar_painel = len(feats_sel_cols) > 0 and df_raw is not None
            r2_panel = None
            if usar_painel and len(df_prev) > 0:
                socio = df_raw[['municipio_norm'] + feats_sel_cols].dropna()
                df_panel = df_base.merge(socio, on='municipio_norm', how='inner')
                X_cols = ['Ano'] + feats_sel_cols
                df_panel_clean = df_panel[X_cols + [prev_nota_col]].dropna()
                if len(df_panel_clean) >= 10:
                    scaler_p = StandardScaler()
                    X_sc = scaler_p.fit_transform(df_panel_clean[X_cols])
                    model_p = Ridge(alpha=1.0)
                    model_p.fit(X_sc, df_panel_clean[prev_nota_col].values)
                    r2_panel = round(model_p.score(X_sc, df_panel_clean[prev_nota_col].values), 3)
                    # Previsões 2025 para cada município
                    socio_all = df_raw[['municipio_norm', 'NO_MUNICIPIO_PROVA'] + feats_sel_cols].dropna()
                    X_2025 = socio_all[feats_sel_cols].copy()
                    X_2025.insert(0, 'Ano', ANO_PREV)
                    socio_all = socio_all.copy()
                    socio_all['Previsão 2025 (painel)'] = model_p.predict(
                        scaler_p.transform(X_2025)).round(2)
                    df_prev = df_prev.merge(
                        socio_all[['NO_MUNICIPIO_PROVA', 'Previsão 2025 (painel)']],
                        on='NO_MUNICIPIO_PROVA', how='left')
                else:
                    usar_painel = False

            df_prev = df_prev.sort_values('Previsão 2025 (tendência)', ascending=False).reset_index(drop=True)

            # ── Seletor de município (só municípios com previsão) ──
            # Fix: usar apenas nomes de df_prev para evitar nomes uppercase de 2012-2014
            muns_prev = sorted(df_prev['Município'].dropna().unique())
            with prev_col2:
                mun_prev_sel = st.selectbox(
                    "Município a visualizar:",
                    options=['Espírito Santo (média geral)'] + muns_prev,
                    key='prev_mun_sel',
                )

            # ── 3. Dados do gráfico para município ou ES ───────────
            if mun_prev_sel == 'Espírito Santo (média geral)':
                serie_graf = df_base.groupby('Ano')[prev_nota_col].mean().reset_index()
                serie_graf.columns = ['Ano', 'Nota']
                title_graf = f"Previsão 2025 — {prev_nota} (média ES)"
                sl_g, ic_g, r_g, _, se_g = stats.linregress(serie_graf['Ano'], serie_graf['Nota'])
                n_g, xm_g = len(serie_graf), serie_graf['Ano'].mean()
                ssx_g = ((serie_graf['Ano'] - xm_g) ** 2).sum()
                se_pg = se_g * np.sqrt(1 + 1/n_g + (ANO_PREV - xm_g)**2 / ssx_g)
                tc_g  = t_dist.ppf(0.975, df=n_g - 2)
                y_temp = round(sl_g * ANO_PREV + ic_g, 1)
                lo_g   = round(y_temp - tc_g * se_pg, 1)
                hi_g   = round(y_temp + tc_g * se_pg, 1)
                r2_g   = round(r_g**2, 3)
                # Painel ES = média das previsões por município
                y_panel = round(df_prev['Previsão 2025 (painel)'].mean(), 1) \
                    if usar_painel and 'Previsão 2025 (painel)' in df_prev.columns else None
            else:
                row_m = df_prev[df_prev['Município'] == mun_prev_sel]
                grp_graf = df_base[df_base['NO_MUNICIPIO_PROVA'] == mun_prev_sel]
                serie_graf = grp_graf[['Ano', prev_nota_col]].dropna().sort_values('Ano')
                serie_graf.columns = ['Ano', 'Nota']
                title_graf = f"Previsão 2025 — {prev_nota} ({mun_prev_sel})"
                sl_g       = row_m['_slope'].values[0]
                ic_g       = row_m['_intercept'].values[0]
                y_temp     = row_m['Previsão 2025 (tendência)'].values[0]
                lo_g       = row_m['IC inferior (95%)'].values[0]
                hi_g       = row_m['IC superior (95%)'].values[0]
                r2_g       = row_m['R² temporal'].values[0]
                y_panel    = row_m['Previsão 2025 (painel)'].values[0] \
                    if usar_painel and 'Previsão 2025 (painel)' in row_m.columns else None

            # Ponto principal = painel (se ativo), senão temporal
            y_main  = y_panel if (usar_painel and y_panel is not None and pd.notna(y_panel)) else y_temp
            label_main = 'Previsão 2025 (painel + tendência)' if usar_painel and y_panel is not None else 'Previsão 2025 (tendência)'

            # ── 4. Gráfico ─────────────────────────────────────────
            with prev_col1:
                anos_ext   = sorted(set(list(serie_graf['Ano']) + [ANO_PREV]))
                trend_vals = [sl_g * a + ic_g for a in anos_ext]

                fig_prev = px.scatter(serie_graf, x='Ano', y='Nota',
                    title=title_graf, template="plotly_dark",
                    labels={'Nota': prev_nota, 'Ano': 'Ano'})
                fig_prev.update_traces(marker=dict(size=9, color='#38BDF8'), name='Real')
                fig_prev.add_scatter(x=anos_ext, y=trend_vals, mode='lines',
                    name='Tendência (OLS)',
                    line=dict(color='#94A3B8', dash='dot', width=2))
                fig_prev.add_scatter(x=[ANO_PREV], y=[y_temp], mode='markers',
                    name='Previsão 2025 (tendência)',
                    marker=dict(size=10 if usar_painel else 14,
                                color='#F97316' if not usar_painel else '#64748b',
                                symbol='diamond', opacity=0.6 if usar_painel else 1.0),
                    error_y=dict(type='data', symmetric=False,
                                 array=[hi_g - y_temp], arrayminus=[y_temp - lo_g],
                                 color='#F97316' if not usar_painel else '#64748b',
                                 thickness=2, width=6) if not usar_painel else None,
                    hovertemplate=(
                        f"<b>Previsão 2025 — tendência</b><br>"
                        f"{prev_nota}: <b>{y_temp:.1f}</b><br>"
                        f"IC 95%: [{lo_g:.1f} – {hi_g:.1f}]<extra></extra>"
                    ))
                if usar_painel and y_panel is not None and pd.notna(y_panel):
                    fig_prev.add_scatter(x=[ANO_PREV], y=[y_panel], mode='markers',
                        name='Previsão 2025 (painel)',
                        marker=dict(size=16, color='#A78BFA', symbol='star'),
                        hovertemplate=(
                            f"<b>Previsão 2025 — painel</b><br>"
                            f"{prev_nota}: <b>{y_panel:.1f}</b><extra></extra>"
                        ))
                fig_prev.add_vrect(x0=2019.5, x1=2022.5,
                    fillcolor="rgba(251,191,36,0.07)", layer="below", line_width=0)
                fig_prev.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E2E8F0"), height=400,
                    xaxis=dict(tickmode='array', tickvals=anos_ext),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.28))
                st.plotly_chart(fig_prev, use_container_width=True)

            # ── Métricas ───────────────────────────────────────────
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Previsão principal 2025", f"{y_main:.1f}",
                       help="Painel (indicadores + tendência) quando features ativas, senão só tendência.")
            mc2.metric("IC 95% (tendência)", f"{lo_g:.1f} – {hi_g:.1f}")
            mc3.metric("Tendência anual", f"{sl_g:+.2f} pts/ano")
            mc4.metric("R² temporal", f"{r2_g:.3f}")
            if usar_painel and r2_panel:
                st.caption(f"Modelo de painel — R² = {r2_panel} | "
                           f"features: Ano + {', '.join(feats_sel_labels)}")

            # ── Tabela comparativa ─────────────────────────────────
            st.markdown("**Previsão 2025 por município:**")
            cols_base = ['Município', f'Nota {anos_base[-1]}',
                         'Previsão 2025 (tendência)', 'IC inferior (95%)', 'IC superior (95%)',
                         'Tendência anual', 'R² temporal']
            if usar_painel and 'Previsão 2025 (painel)' in df_prev.columns:
                cols_base.insert(3, 'Previsão 2025 (painel)')
            cols_show = [c for c in cols_base if c in df_prev.columns]
            st.dataframe(df_prev[cols_show], use_container_width=True, hide_index=True)

# ─── TAB 5: EVOLUÇÃO TEMPORAL ────────────────────────────────────
with tab5:
    st.markdown("### :material/trending_up: Evolução Temporal das Notas — ENEM 2012–2024")
    st.markdown(
        "Análise da evolução das notas médias no ENEM ao longo de 13 anos "
        "nos municípios do Espírito Santo. Permite identificar tendências, "
        "o impacto da pandemia (2020–2022) e comparar perfis de desenvolvimento."
    )

    if df_temporal is None:
        st.warning(
            erro_temporal or
            "Arquivo **dados_temporal.csv** não encontrado. Execute `python process_temporal.py` para gerar os dados históricos."
        )
        st.code("python process_temporal.py", language="bash")
        st.stop()

    # Carregar IDHM de referência para agrupar por perfil
    @st.cache_data
    def load_idhm_ref():
        for f in ('dados_compilados_v4.csv', 'dados_compilados.csv',
                  'dados_publico.csv', 'dados_todos.csv'):
            try:
                tmp = pd.read_csv(f)
                if 'municipio_norm' in tmp.columns and 'IDHM' in tmp.columns:
                    return tmp[['municipio_norm', 'Nome_Municipio', 'IDHM']].dropna()
            except FileNotFoundError:
                continue
        return None

    idhm_ref = load_idhm_ref()

    df_t = df_temporal.copy()
    df_t['Ano'] = df_t['Ano'].astype(int)

    anos_disponiveis = sorted(df_t['Ano'].unique())
    muns_disponiveis = sorted(df_t['NO_MUNICIPIO_PROVA'].dropna().unique())

    # ── SEÇÃO 1: Evolução por município selecionado ───────────────
    st.markdown("---")
    st.markdown("#### :material/location_city: Comparar Municípios ao Longo do Tempo")

    # Municípios com dados em pelo menos 8 anos (série mais completa)
    muns_completos = (
        df_t.groupby('NO_MUNICIPIO_PROVA')['Ano'].count()
        .loc[lambda x: x >= 8].index.tolist()
    )

    # 4 representantes por quartil de IDHM (maior volume de candidatos de cada faixa)
    QUARTIL_LABELS = {
        'Q1': ('🔴', 'Baixo IDHM',    'Interior mais vulnerável — menor IDH, maior dependência do ensino público'),
        'Q2': ('🟠', 'Médio-baixo',   'Municípios do interior com infraestrutura educacional em desenvolvimento'),
        'Q3': ('🟡', 'Médio-alto',    'Cidades médias com boa rede de ensino público'),
        'Q4': ('🟢', 'Alto IDHM',     'Municípios mais desenvolvidos — Grande Vitória e cidades-polo'),
    }

    idhm_q = None
    muns_default = []
    quartil_map = {}  # municipio_norm → quartil label

    if idhm_ref is not None and len(idhm_ref) > 4:
        idhm_q = idhm_ref.copy()
        idhm_q['Quartil'] = pd.qcut(idhm_q['IDHM'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

        muns_completos_norm = (
            df_t[df_t['NO_MUNICIPIO_PROVA'].isin(muns_completos)]
            [['NO_MUNICIPIO_PROVA', 'municipio_norm']].drop_duplicates()
        )
        cand_total = (
            df_t[df_t['NO_MUNICIPIO_PROVA'].isin(muns_completos)]
            .groupby('municipio_norm')['QTD_CANDIDATOS'].sum()
            .reset_index()
        )
        merged = (
            idhm_q.merge(cand_total, on='municipio_norm', how='inner')
                  .merge(muns_completos_norm, on='municipio_norm', how='inner')
                  .sort_values('QTD_CANDIDATOS', ascending=False)
        )
        # Top 3 por quartil (12 municípios = equilíbrio entre diversidade e legibilidade)
        top4 = merged.groupby('Quartil', observed=True).head(3).reset_index(drop=True)
        muns_default = top4['NO_MUNICIPIO_PROVA'].tolist()

        # Mapa município → quartil para legenda
        quartil_map = dict(zip(merged['NO_MUNICIPIO_PROVA'], merged['Quartil'].astype(str)))

        # Legenda dos quartis
        st.markdown("**Classificação dos municípios por IDHM (Atlas Brasil 2010):**")
        lcols = st.columns(4)
        for i, (qk, (emoji, nome, desc)) in enumerate(QUARTIL_LABELS.items()):
            muns_q = merged[merged['Quartil'] == qk]['NO_MUNICIPIO_PROVA'].tolist()
            faixa  = idhm_q[idhm_q['Quartil'] == qk]['IDHM']
            with lcols[i]:
                st.markdown(
                    f'<div class="stat-box"><b>{emoji} {qk} — {nome}</b><br>'
                    f'<span style="color:#64748b;font-size:0.8rem">{desc}</span><br><br>'
                    f'<span style="color:#94A3B8;font-size:0.78rem">IDHM: {faixa.min():.3f} – {faixa.max():.3f}</span><br>'
                    f'<span style="color:#94A3B8;font-size:0.78rem">{len(muns_q)} municípios</span></div>',
                    unsafe_allow_html=True
                )
        st.markdown("")
    else:
        muns_default = (
            df_t[df_t['NO_MUNICIPIO_PROVA'].isin(muns_completos)]
            .groupby('NO_MUNICIPIO_PROVA')['QTD_CANDIDATOS'].sum()
            .nlargest(8).index.tolist()
        )

    def _fmt_mun(m):
        q = quartil_map.get(m, '')
        if q:
            emoji = QUARTIL_LABELS.get(q, ('', '', ''))[0]
            return f"{emoji} {m} ({q})"
        return m

    muns_sel = st.multiselect(
        "Selecione os municípios:",
        options=muns_completos,
        default=[m for m in muns_default if m in muns_completos],
        format_func=_fmt_mun,
        key='temp_muns',
        help="Apenas municípios com dados em ≥ 8 anos. Padrão: top 3 de cada quartil de IDHM.",
    )

    nota_sel = st.selectbox(
        "Nota a visualizar:",
        options={
            'Nota Média Geral':    'NOTA_MEDIA',
            'Matemática':          'NU_NOTA_MT',
            'Redação':             'NU_NOTA_REDACAO',
            'Linguagens':          'NU_NOTA_LC',
            'Ciências Humanas':    'NU_NOTA_CH',
            'Ciências da Natureza':'NU_NOTA_CN',
        }.keys(),
        key='temp_nota',
    )
    nota_col = {
        'Nota Média Geral':    'NOTA_MEDIA',
        'Matemática':          'NU_NOTA_MT',
        'Redação':             'NU_NOTA_REDACAO',
        'Linguagens':          'NU_NOTA_LC',
        'Ciências Humanas':    'NU_NOTA_CH',
        'Ciências da Natureza':'NU_NOTA_CN',
    }[nota_sel]

    if muns_sel:
        df_sel = df_t[df_t['NO_MUNICIPIO_PROVA'].isin(muns_sel)].copy()
        # Adicionar label de quartil ao nome do município para a legenda
        if quartil_map:
            df_sel['Município'] = df_sel['NO_MUNICIPIO_PROVA'].map(
                lambda m: f"{QUARTIL_LABELS.get(quartil_map.get(m,''), ('','',''))[0]} {m} ({quartil_map.get(m,'')})"
            )
        else:
            df_sel['Município'] = df_sel['NO_MUNICIPIO_PROVA']

        fig_line = px.line(
            df_sel, x='Ano', y=nota_col,
            color='Município',
            markers=True,
            title=f"{nota_sel} — Evolução por município (2012–2024)",
            labels={'Município': 'Município', nota_col: nota_sel, 'Ano': 'Ano'},
            template="plotly_dark",
        )
        # Destacar período da pandemia (2020–2022: escolas fechadas e recuperação)
        fig_line.add_vrect(
            x0=2019.5, x1=2022.5,
            fillcolor="rgba(251,191,36,0.08)",
            layer="below", line_width=0,
            annotation_text="Pandemia (2020–2022)", annotation_position="top left",
            annotation_font_color="#FBB724",
        )
        anos_grafico = [a for a in anos_disponiveis if a >= 2015]
        fig_line.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"), height=480,
            xaxis=dict(
                tickmode='array', tickvals=anos_grafico,
                range=[2014.5, anos_disponiveis[-1] + 0.5],
            ),
        )
        st.plotly_chart(fig_line, use_container_width=True)
        st.caption("Faixa amarela = período pandêmico (2020–2022). Série exibida a partir de 2015 — anos anteriores têm cobertura insuficiente de municípios.")
    else:
        st.info("Selecione ao menos um município para visualizar a evolução.")

    # ── SEÇÃO 2: Evolução por perfil de desenvolvimento ──────────
    st.markdown("---")
    st.markdown("#### :material/holiday_village: Evolução Média por Perfil de Desenvolvimento (IDHM)")
    st.markdown(
        "Municípios agrupados em 4 quartis de IDHM (dados Atlas Brasil 2010). "
        "Permite identificar se a desigualdade de desempenho aumentou ou diminuiu ao longo dos anos "
        "e como a pandemia afetou cada perfil de forma diferente."
    )

    if idhm_ref is not None and len(idhm_ref) > 10:
        # Atribuir quartil de IDHM
        idhm_ref = idhm_ref.copy()
        idhm_ref['Perfil'] = pd.qcut(
            idhm_ref['IDHM'], q=4,
            labels=['Q1 — Baixo IDHM', 'Q2 — Médio-baixo', 'Q3 — Médio-alto', 'Q4 — Alto IDHM']
        )

        df_t2 = df_t.merge(
            idhm_ref[['municipio_norm', 'Perfil']],
            on='municipio_norm', how='left'
        ).dropna(subset=['Perfil'])

        if len(df_t2) > 0:
            perfil_anual = df_t2.groupby(['Ano', 'Perfil'])[nota_col].mean().reset_index()
            perfil_anual.columns = ['Ano', 'Perfil', nota_sel]

            fig_perfil = px.line(
                perfil_anual, x='Ano', y=nota_sel,
                color='Perfil',
                markers=True,
                title=f"{nota_sel} média por quartil de IDHM — 2012 a 2024",
                labels={'Ano': 'Ano', nota_sel: nota_sel},
                template="plotly_dark",
                color_discrete_sequence=['#EF4444', '#F97316', '#38BDF8', '#22C55E'],
            )
            fig_perfil.add_vrect(
                x0=2019.5, x1=2021.5,
                fillcolor="rgba(251,191,36,0.08)",
                layer="below", line_width=0,
                annotation_text="Pandemia", annotation_position="top left",
                annotation_font_color="#FBB724",
            )
            fig_perfil.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0"), height=460,
                xaxis=dict(tickmode='array', tickvals=anos_disponiveis),
                legend=dict(title="Perfil", orientation="h", yanchor="bottom", y=-0.25),
            )
            st.plotly_chart(fig_perfil, use_container_width=True)

            # Análise do impacto da pandemia (2019 → 2022)
            st.markdown("**Impacto da pandemia por perfil (variação 2019 → 2022):**")
            anos_pan = perfil_anual[perfil_anual['Ano'].isin([2019, 2022])]
            if len(anos_pan) > 0:
                pivot_pan = anos_pan.pivot(index='Perfil', columns='Ano', values=nota_sel)
                if 2019 in pivot_pan.columns and 2022 in pivot_pan.columns:
                    pivot_pan['Variação (pontos)'] = (pivot_pan[2022] - pivot_pan[2019]).round(2)
                    pivot_pan = pivot_pan.rename(columns={2019: 'Nota 2019', 2022: 'Nota 2022'})
                    pivot_pan = pivot_pan[['Nota 2019', 'Nota 2022', 'Variação (pontos)']].reset_index()
                    st.dataframe(pivot_pan, use_container_width=True, hide_index=True)
                    st.caption(
                        "Período 2020–2022: escolas fechadas (2020), reabertura parcial (2021) "
                        "e retorno integral (2022). Se a variação for maior no Q1 (baixo IDHM), "
                        "a pandemia aprofundou as desigualdades."
                    )
        else:
            st.warning("Não foi possível cruzar dados temporais com IDHM.")
    else:
        st.info("Carregue os dados compilados para ver evolução por perfil de IDHM.")

    # ── SEÇÃO 3: Ranking de evolução ─────────────────────────────
    st.markdown("---")
    st.markdown("#### :material/emoji_events: Municípios que Mais Evoluíram / Regrediram")

    # Modo grupo = média de múltiplos anos; modo par = dois anos específicos
    GRUPO_A = [2012, 2014, 2015, 2016, 2017, 2018]   # pré-pandemia (2013 excluído: anomalia TRI)
    GRUPO_B = [2019, 2020, 2021, 2022, 2023, 2024]   # pandemia + pós

    INTERVALOS = {
        "Pós pandemia (2022 → 2024)":       ('par',   2022, 2024),
        "Pré → Pós pandemia (2019 → 2024)": ('par',   2019, 2024),
        "Durante pandemia (2020 → 2022)":   ('par',   2020, 2022),
        "Médias 2012 a 2024":               ('grupo', GRUPO_A, GRUPO_B),
        "Personalizado":                    ('custom', None, None),
    }

    intervalo_sel = st.radio(
        "Período de comparação:",
        options=list(INTERVALOS.keys()),
        index=0,
        horizontal=True,
        key='temp_intervalo',
    )

    modo, val_a, val_b = INTERVALOS[intervalo_sel]

    if modo == 'par':
        ano_ini = min(anos_disponiveis, key=lambda x: abs(x - val_a))
        ano_fim = min(anos_disponiveis, key=lambda x: abs(x - val_b))
        label_a, label_b = str(ano_ini), str(ano_fim)
        nota_a = df_t[df_t['Ano'] == ano_ini].set_index('municipio_norm')[nota_col]
        nota_b = df_t[df_t['Ano'] == ano_fim].set_index('municipio_norm')[nota_col]
        mun_names = df_t[df_t['Ano'] == ano_ini].set_index('municipio_norm')['NO_MUNICIPIO_PROVA']
        titulo_bar = f"Variação da {nota_sel} entre {ano_ini} e {ano_fim}"

    elif modo == 'grupo':
        anos_a = [a for a in val_a if a in anos_disponiveis]
        anos_b = [a for a in val_b if a in anos_disponiveis]
        label_a = f"Média {anos_a[0]}–{anos_a[-1]}"
        label_b = f"Média {anos_b[0]}–{anos_b[-1]}"
        nota_a = df_t[df_t['Ano'].isin(anos_a)].groupby('municipio_norm')[nota_col].mean()
        nota_b = df_t[df_t['Ano'].isin(anos_b)].groupby('municipio_norm')[nota_col].mean()
        mun_names = df_t[df_t['Ano'].isin(anos_a)].drop_duplicates('municipio_norm').set_index('municipio_norm')['NO_MUNICIPIO_PROVA']
        titulo_bar = f"Variação de {nota_sel}: {label_a} → {label_b}"
        st.markdown(
            f'<div class="info-banner">ℹ️ Comparando a <b>média de {len(anos_a)} anos pré-pandemia</b> '
            f'({", ".join(map(str, anos_a))}) com a <b>média de {len(anos_b)} anos pandemia/pós</b> '
            f'({", ".join(map(str, anos_b))}). '
            f'2013 excluído do grupo A por anomalia na escala TRI documentada pelo INEP.</div>',
            unsafe_allow_html=True
        )

    else:  # custom
        rc1, rc2 = st.columns(2)
        with rc1:
            ano_ini = st.select_slider("Ano inicial:", options=anos_disponiveis,
                                       value=anos_disponiveis[0], key='temp_ano_ini')
        with rc2:
            ano_fim = st.select_slider("Ano final:", options=anos_disponiveis,
                                       value=anos_disponiveis[-1], key='temp_ano_fim')
        label_a, label_b = str(ano_ini), str(ano_fim)
        nota_a = df_t[df_t['Ano'] == ano_ini].set_index('municipio_norm')[nota_col]
        nota_b = df_t[df_t['Ano'] == ano_fim].set_index('municipio_norm')[nota_col]
        mun_names = df_t[df_t['Ano'] == ano_ini].set_index('municipio_norm')['NO_MUNICIPIO_PROVA']
        titulo_bar = f"Variação da {nota_sel} entre {ano_ini} e {ano_fim}"

    df_evo = pd.DataFrame({'Nota_A': nota_a, 'Nota_B': nota_b}).dropna()
    df_evo['Município'] = mun_names.reindex(df_evo.index)
    df_evo['Variação (pontos)'] = (df_evo['Nota_B'] - df_evo['Nota_A']).round(2)
    df_evo = df_evo.rename(columns={'Nota_A': label_a, 'Nota_B': label_b})
    df_evo = df_evo.sort_values('Variação (pontos)', ascending=False).reset_index(drop=True)

    colunas_tabela = ['Município', label_a, label_b, 'Variação (pontos)']
    df_evolucao  = df_evo[df_evo['Variação (pontos)'] > 0].head(10)
    df_regressao = df_evo[df_evo['Variação (pontos)'] < 0].tail(10).iloc[::-1]

    col_top, col_bot = st.columns(2)
    with col_top:
        st.markdown(f"**Top 10 — Maior evolução** ({label_a} → {label_b}):")
        if df_evolucao.empty:
            st.info("Nenhum município evoluiu neste período.")
        else:
            st.dataframe(df_evolucao[colunas_tabela], use_container_width=True, hide_index=True)
    with col_bot:
        st.markdown(f"**Top 10 — Maior regressão** ({label_a} → {label_b}):")
        if df_regressao.empty:
            st.success(f"Todos os {len(df_evo)} municípios evoluíram neste período — nenhuma regressão registrada.")
        else:
            st.dataframe(df_regressao[colunas_tabela], use_container_width=True, hide_index=True)

    df_evo_show = pd.concat([df_evo.head(15), df_evo.tail(10)]).drop_duplicates()
    fig_evo = px.bar(
        df_evo_show,
        x='Variação (pontos)', y='Município',
        orientation='h',
        color='Variação (pontos)',
        color_continuous_scale='RdYlGn',
        title=titulo_bar,
        template="plotly_dark",
    )
    fig_evo.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"), height=600,
        coloraxis_showscale=False,
        yaxis={'categoryorder': 'total ascending'},
    )
    fig_evo.add_vline(x=0, line_dash="dash", line_color="#94A3B8")
    st.plotly_chart(fig_evo, use_container_width=True)
