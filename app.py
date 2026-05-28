import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

st.set_page_config(page_title="Desigualdade vs ENEM no ES", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0E1117; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #E2E8F0; font-weight: 700; }
    .metric-card {
        background: rgba(30,41,59,0.7); backdrop-filter: blur(10px);
        border-radius: 12px; padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.1); transition: transform 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-5px); }
    .metric-title { color: #94A3B8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #38BDF8; font-size: 2rem; font-weight: bold; margin-top: 10px; }
    .metric-sub   { color: #64748b; font-size: 0.8rem; margin-top: 5px; }
    .stat-box {
        background: rgba(30,41,59,0.5); border-radius: 8px; padding: 14px;
        border: 1px solid rgba(255,255,255,0.08); margin-top: 8px;
    }
    .filter-banner {
        background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.3);
        border-radius: 10px; padding: 12px 18px; margin-bottom: 16px;
        color: #94A3B8; font-size: 0.9rem;
    }
    .warn-banner {
        background: rgba(251,191,36,0.1); border: 1px solid rgba(251,191,36,0.3);
        border-radius: 10px; padding: 12px 18px; margin-bottom: 8px;
        color: #94A3B8; font-size: 0.85rem;
    }
    .info-banner {
        background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.3);
        border-radius: 10px; padding: 12px 18px; margin-bottom: 8px;
        color: #94A3B8; font-size: 0.85rem;
    }
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
st.title("🎓 Desigualdade Social e Desempenho no ENEM — Espírito Santo 2024")
st.markdown(
    "Análise exploratória das notas do **ENEM 2024** nos municípios do ES, "
    "cruzadas com indicadores socioeconômicos (Atlas Brasil / Censo 2010) "
    "e gastos educacionais municipais (SIOPE 2023)."
)

tipo_escola = st.radio(
    "🏫 Candidatos incluídos:",
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

@st.cache_data
def load_data(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

df_raw = load_data(arquivo)

if df_raw is None:
    st.error(f"Arquivo **{arquivo}** não encontrado. Execute `process_duplo.py` para gerar os dados.")
    st.stop()

st.markdown("---")
min_cand = st.slider(
    "👥 Mínimo de candidatos por município:",
    min_value=10, max_value=300, value=st.session_state['min_cand'], step=10,
    key="min_cand",
    help="Municípios com poucos candidatos têm nota média instável e podem distorcer correlações."
)

df = df_raw[df_raw['QTD_CANDIDATOS'] >= min_cand].copy().reset_index(drop=True)
excluidos = len(df_raw) - len(df)

if excluidos > 0:
    st.markdown(
        f'<div class="warn-banner">⚠️ <b>{excluidos} município(s) excluído(s)</b> por ter menos de {min_cand} candidatos. '
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
    st.markdown(f'<div class="metric-card"><div class="metric-title">Municípios Analisados</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Total de Candidatos</div><div class="metric-value">{int(df["QTD_CANDIDATOS"].sum()):,}</div></div>', unsafe_allow_html=True)
with c3:
    idx = df["NOTA_MEDIA"].idxmax()
    st.markdown(f'<div class="metric-card"><div class="metric-title">Maior Nota Média</div><div class="metric-value">{df.loc[idx,"NOTA_MEDIA"]:.1f}</div><div class="metric-sub">{df.loc[idx,"Nome_Municipio"]}</div></div>', unsafe_allow_html=True)
with c4:
    idx = df["NOTA_MEDIA"].idxmin()
    st.markdown(f'<div class="metric-card"><div class="metric-title">Menor Nota Média</div><div class="metric-value">{df.loc[idx,"NOTA_MEDIA"]:.1f}</div><div class="metric-sub">{df.loc[idx,"Nome_Municipio"]}</div></div>', unsafe_allow_html=True)

# ── DADOS TEMPORAIS (carregados uma vez, sem depender do filtro de escola) ──
@st.cache_data
def load_temporal():
    try:
        return pd.read_csv("dados_temporal.csv")
    except FileNotFoundError:
        return None

df_temporal = load_temporal()

# ── ABAS ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Explorador de Correlação",
    "🧪 Testes Estatísticos",
    "🎯 Outliers e Distribuição",
    "🤖 Machine Learning",
    "📈 Evolução Temporal",
])

# ─── TAB 1 ────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 📊 Explorador de Correlação")
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
                f'<div class="info-banner">ℹ️ <b>% de Impostos Aplicados em Educação:</b> '
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
    st.markdown("### 🗃️ Dados por Município")
    disp = [c for c in ['Nome_Municipio', 'QTD_CANDIDATOS', 'PERC_ESCOLA_PUB', 'NOTA_MEDIA',
                         'IDHM', 'IDHM_R', 'IDHM_E', 'GINI', 'TX_ANALF', 'NEET_VULN',
                         'IVCAD', 'IVCAD_DR', 'IVCAD_TQA',
                         'EDU_Perc_Aplicacao', 'EDU_Investimento_Aluno', 'EDU_Aplicacao_Total']
            if c in df.columns]
    st.dataframe(df[disp].sort_values('NOTA_MEDIA', ascending=False).reset_index(drop=True),
                 use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🌡️ Matriz de Correlação entre Indicadores")
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
    st.markdown("#### 📈 Regressão Linear Simples")
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
    st.markdown("#### 🔬 Teste t — Comparação de Grupos")
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
    st.markdown("#### 📋 Tabela de Correlações com a Nota Média Geral")
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
    st.markdown("### 🎯 Detecção de Outliers e Distribuição")

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
    st.markdown("### 🤖 Machine Learning — Agrupamento e Importância de Variáveis")
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
    st.markdown("#### 🎯 K-Means — Agrupamento de Municípios em Perfis")
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
    st.markdown("#### 🌲 Random Forest — Importância das Variáveis")
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
    st.markdown("#### 🔮 Previsão para o ENEM 2025")
    st.markdown(
        "Regressão linear temporal aplicada à série histórica 2012–2024 de cada município. "
        "A tendência é extrapolada para 2025 com intervalo de confiança de 95%."
    )
    st.markdown(
        '<div class="warn-banner">⚠️ <b>Limitação metodológica:</b> previsão baseada em tendência linear com ~9 pontos por município. '
        'Fatores externos (mudanças no ENEM, políticas educacionais, eventos imprevistos) não são capturados. '
        'Use como estimativa de tendência, não como previsão precisa.</div>',
        unsafe_allow_html=True
    )

    if df_temporal is None:
        st.info("Execute `python process_temporal.py` para habilitar a previsão 2025.")
    else:
        from scipy.stats import t as t_dist

        df_t_ml = df_temporal.copy()
        df_t_ml['Ano'] = df_t_ml['Ano'].astype(int)
        anos_disp_ml = sorted(df_t_ml['Ano'].unique())

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

        if len(anos_base) < 3:
            st.warning("Selecione ao menos 3 anos para ajustar o modelo.")
        else:
            ANO_PREV = 2025
            df_base = df_t_ml[df_t_ml['Ano'].isin(anos_base)].copy()

            previsoes = []
            for mun, grp in df_base.groupby('NO_MUNICIPIO_PROVA'):
                serie = grp[['Ano', prev_nota_col]].dropna().sort_values('Ano')
                if len(serie) < 3:
                    continue
                x = serie['Ano'].values
                y = serie[prev_nota_col].values
                slope, intercept, r, p, se = stats.linregress(x, y)
                n      = len(x)
                x_mean = x.mean()
                ss_x   = ((x - x_mean) ** 2).sum()
                se_pred = se * np.sqrt(1 + 1/n + (ANO_PREV - x_mean)**2 / ss_x)
                t_crit  = t_dist.ppf(0.975, df=n - 2)
                y_pred  = slope * ANO_PREV + intercept
                previsoes.append({
                    'Município':         mun,
                    f'Nota {anos_base[-1]} (real)': round(
                        grp[grp['Ano'] == anos_base[-1]][prev_nota_col].values[0]
                        if anos_base[-1] in grp['Ano'].values else np.nan, 2),
                    'Previsão 2025':     round(y_pred, 2),
                    'IC inferior (95%)': round(y_pred - t_crit * se_pred, 2),
                    'IC superior (95%)': round(y_pred + t_crit * se_pred, 2),
                    'Tendência anual':   round(slope, 2),
                    'R²':                round(r**2, 3),
                })

            df_prev = pd.DataFrame(previsoes).sort_values('Previsão 2025', ascending=False).reset_index(drop=True)

            # Previsão agregada ES
            es_anual = df_base.groupby('Ano')[prev_nota_col].mean().reset_index()
            es_anual.columns = ['Ano', 'Nota']
            x_es = es_anual['Ano'].values
            y_es = es_anual['Nota'].values
            sl_es, ic_es, r_es, p_es, se_es = stats.linregress(x_es, y_es)
            n_es     = len(x_es)
            xm_es    = x_es.mean()
            ssx_es   = ((x_es - xm_es) ** 2).sum()
            se_p_es  = se_es * np.sqrt(1 + 1/n_es + (ANO_PREV - xm_es)**2 / ssx_es)
            t_es     = t_dist.ppf(0.975, df=n_es - 2)
            prev_es  = sl_es * ANO_PREV + ic_es
            prev_es_lo = prev_es - t_es * se_p_es
            prev_es_hi = prev_es + t_es * se_p_es

            with prev_col1:
                anos_ext   = list(anos_base) + [ANO_PREV]
                trend_line = [sl_es * a + ic_es for a in anos_ext]
                fig_prev = px.scatter(
                    es_anual, x='Ano', y='Nota',
                    title=f"Tendência e previsão 2025 — {prev_nota} (média ES)",
                    template="plotly_dark",
                    labels={'Nota': prev_nota, 'Ano': 'Ano'},
                )
                fig_prev.update_traces(marker=dict(size=9, color='#38BDF8'), name='Real')
                fig_prev.add_scatter(
                    x=anos_ext, y=trend_line,
                    mode='lines', name='Tendência (OLS)',
                    line=dict(color='#94A3B8', dash='dot', width=2),
                )
                fig_prev.add_scatter(
                    x=[ANO_PREV], y=[prev_es],
                    mode='markers', name='Previsão 2025',
                    marker=dict(size=14, color='#F97316', symbol='diamond'),
                    error_y=dict(type='data', symmetric=False,
                                 array=[prev_es_hi - prev_es],
                                 arrayminus=[prev_es - prev_es_lo],
                                 color='#F97316', thickness=2, width=6),
                    hovertemplate=(
                        f"<b>Previsão 2025</b><br>"
                        f"{prev_nota}: <b>{prev_es:.1f}</b><br>"
                        f"IC 95%: [{prev_es_lo:.1f} – {prev_es_hi:.1f}]"
                        "<extra></extra>"
                    ),
                )
                fig_prev.add_vrect(
                    x0=2019.5, x1=2022.5,
                    fillcolor="rgba(251,191,36,0.07)", layer="below", line_width=0,
                )
                fig_prev.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E2E8F0"), height=380,
                    xaxis=dict(tickmode='array', tickvals=anos_ext),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25),
                )
                st.plotly_chart(fig_prev, use_container_width=True)

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Previsão ES 2025",  f"{prev_es:.1f}")
            mc2.metric("IC 95% inferior",   f"{prev_es_lo:.1f}")
            mc3.metric("IC 95% superior",   f"{prev_es_hi:.1f}")
            mc4.metric("Tendência anual",   f"{sl_es:+.2f} pts/ano")
            st.caption(f"R² do modelo ES = {r_es**2:.3f} | p-valor = {p_es:.4f} | n = {n_es} anos")

            st.markdown("**Previsão 2025 por município:**")
            cols_show = ['Município', f'Nota {anos_base[-1]} (real)', 'Previsão 2025',
                         'IC inferior (95%)', 'IC superior (95%)', 'Tendência anual', 'R²']
            st.dataframe(df_prev[cols_show], use_container_width=True, hide_index=True)

# ─── TAB 5: EVOLUÇÃO TEMPORAL ────────────────────────────────────
with tab5:
    st.markdown("### 📈 Evolução Temporal das Notas — ENEM 2012–2024")
    st.markdown(
        "Análise da evolução das notas médias no ENEM ao longo de 13 anos "
        "nos municípios do Espírito Santo. Permite identificar tendências, "
        "o impacto da pandemia (2020–2022) e comparar perfis de desenvolvimento."
    )

    if df_temporal is None:
        st.warning(
            "Arquivo **dados_temporal.csv** não encontrado. "
            "Execute `python process_temporal.py` para gerar os dados históricos."
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
    st.markdown("#### 🏙️ Comparar Municípios ao Longo do Tempo")

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
    st.markdown("#### 🏘️ Evolução Média por Perfil de Desenvolvimento (IDHM)")
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
    st.markdown("#### 🏆 Municípios que Mais Evoluíram / Regrediram")

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

