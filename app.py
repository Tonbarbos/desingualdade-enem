import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

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
    'tipo_escola': "Apenas escola pública",
    'min_cand':    30,
    'cat_x':       'DESIGUALDADE (Censo 2010)',
    'cat_y':       'NOTAS DO ENEM',
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

f1, f2 = st.columns([2, 3])
with f1:
    tipo_escola = st.radio(
        "🏫 Candidatos incluídos:",
        ["Apenas escola pública", "Pública + Privada (todos)"],
        horizontal=True,
        key="tipo_escola",
    )
with f2:
    if tipo_escola == "Apenas escola pública":
        st.markdown('<div class="filter-banner">📌 <b>Escola pública</b> (Federal, Estadual e Municipal). Recomendado para comparações entre municípios — elimina o viés de concentração de escolas privadas em municípios mais ricos.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="filter-banner">⚠️ <b>Todos os candidatos.</b> Municípios com mais escolas privadas tendem a ter notas médias mais altas, o que pode distorcer correlações socioeconômicas.</div>', unsafe_allow_html=True)

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
    min_value=10, max_value=300, step=10,
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
    'DESIGUALDADE (Censo 2010)': {
        'Índice de Gini':                          'GINI',
        'Taxa de Analfabetismo (15+ anos)':        'TX_ANALF',
        '% Crianças em Dom. Sem Ens. Fundamental': 'CRIAN_VULN',
        '% 15-24 anos Nem Estuda Nem Trabalha':    'NEET_VULN',
        '% de 15-17 ainda no Ensino Fundamental':  'T_FREQ1517_FUND',
    },
    'EDUCAÇÃO (Censo 2010)': {
        'Expectativa de Anos de Estudo (18 anos)':      'E_ANOSESTUDO',
        '% com Ensino Fundamental Completo (25+ anos)': 'PERC_FUND_COMP',
        '% com Ensino Médio Completo (25+ anos)':       'PERC_MED_COMP',
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

# ── ABAS ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Explorador de Correlação",
    "🌡️ Matriz de Correlação",
    "🧪 Testes Estatísticos",
    "🎯 Outliers e Distribuição",
])

# ─── TAB 1 ────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 📊 Explorador de Correlação")
    st.markdown("Selecione dois indicadores para visualizar a relação entre eles. O tamanho dos pontos representa o número de candidatos.")

    cx, cy = st.columns(2)
    with cx:
        st.markdown("<p style='color:#94A3B8;margin-bottom:0'>Eixo X (variável independente):</p>", unsafe_allow_html=True)
        # FIX 1: usa session_state para preservar seleção ao trocar filtro de escola
        cat_x_idx = list(INDICATORS_DICT.keys()).index(st.session_state['cat_x']) \
                    if st.session_state['cat_x'] in INDICATORS_DICT else 0
        cat_x   = st.selectbox("Cat X:", list(INDICATORS_DICT.keys()),
                                index=cat_x_idx, key='cat_x', label_visibility="collapsed")
        opts_x  = {k: v for k, v in INDICATORS_DICT[cat_x].items()
                   if v in df.columns and df[v].count() > 1}
        label_x = st.selectbox("Ind X:", list(opts_x.keys()), key='ind_x', label_visibility="collapsed")
        x_col   = opts_x[label_x]

    with cy:
        st.markdown("<p style='color:#94A3B8;margin-bottom:0'>Eixo Y (variável dependente):</p>", unsafe_allow_html=True)
        cats_y      = [c for c in INDICATORS_DICT.keys() if c != cat_x]
        cat_y_default = st.session_state['cat_y'] if st.session_state['cat_y'] in cats_y else cats_y[0]
        default_idx = cats_y.index(cat_y_default)
        cat_y       = st.selectbox("Cat Y:", cats_y, index=default_idx,
                                   key='cat_y', label_visibility="collapsed")

        opts_y = {}
        for lbl, col in INDICATORS_DICT[cat_y].items():
            if col not in df.columns or df[col].count() <= 1: continue
            r = df[x_col].corr(df[col])
            if pd.isna(r): continue
            arrow, strength = corr_label(r, 1)  # p não importa aqui, só direção
            opts_y[f"{arrow} {lbl} ({strength})"] = col

        if not opts_y:
            st.warning("Sem dados correlacionáveis nesta categoria.")
            st.stop()

        label_y_full = st.selectbox("Ind Y:", list(opts_y.keys()), key='ind_y', label_visibility="collapsed")
        y_col        = opts_y[label_y_full]
        y_axis_label = label_y_full.split('(')[0][2:].strip()

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
        st.markdown(
            '<div class="warn-banner">💡 <b>Correlação fraca</b> pode ter causas metodológicas: '
            '(1) filtro de escola pública homogeneíza as notas, comprimindo a variação entre municípios; '
            '(2) filtro de candidatos mínimos muito alto reduz a amostra e aumenta instabilidade; '
            '(3) a relação pode não ser linear. Correlações fracas também são resultados válidos e discutíveis no trabalho.</div>',
            unsafe_allow_html=True
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
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🗃️ Dados por Município")
    disp = [c for c in ['Nome_Municipio', 'QTD_CANDIDATOS', 'PERC_ESCOLA_PUB', 'NOTA_MEDIA',
                         'IDHM', 'IDHM_R', 'IDHM_E', 'GINI', 'TX_ANALF', 'NEET_VULN',
                         'EDU_Perc_Aplicacao', 'EDU_Investimento_Aluno', 'EDU_Aplicacao_Total']
            if c in df.columns]
    st.dataframe(df[disp].sort_values('NOTA_MEDIA', ascending=False).reset_index(drop=True),
                 use_container_width=True, hide_index=True)

# ─── TAB 2 ────────────────────────────────────────────────────────
with tab2:
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

# ─── TAB 3 ────────────────────────────────────────────────────────
with tab3:
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

# ─── TAB 4 ────────────────────────────────────────────────────────
with tab4:
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
        show_cols = [c for c in ['Nome_Municipio', out_col, 'NOTA_MEDIA', 'IDHM', 'QTD_CANDIDATOS'] if c in outliers.columns]
        show = list(dict.fromkeys(show_cols)) # Remove duplicates while preserving order
        st.dataframe(
            outliers[show].reset_index(drop=True).sort_values(out_col, ascending=False).reset_index(drop=True),
            use_container_width=True, hide_index=True
        )
