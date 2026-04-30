import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuração da Página
st.set_page_config(page_title="Desigualdade vs ENEM no ES", page_icon="🎓", layout="wide")

# CSS para Premium Design
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #E2E8F0;
        font-weight: 700;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        color: #38BDF8;
        font-size: 2rem;
        font-weight: bold;
        margin-top: 10px;
    }
    .metric-sub {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dados_compilados.csv")
        return df
    except FileNotFoundError:
        return None

df = load_data()

st.title("🎓 Desigualdade Social e Desempenho no ENEM — Espírito Santo 2024")
st.markdown("Análise exploratória das notas do **ENEM 2024** nos 78 municípios do ES, cruzadas com indicadores socioeconômicos (Atlas Brasil / Censo 2010) e gastos educacionais municipais (SIOPE 2023).")
st.markdown("---")

if df is None:
    st.error("Arquivo **dados_compilados.csv** não encontrado.")
    st.stop()

# ── OVERVIEW METRICS ──────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Municípios Analisados</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
with col2:
    total = df["QTD_CANDIDATOS"].sum()
    st.markdown(f'<div class="metric-card"><div class="metric-title">Total de Candidatos (ES)</div><div class="metric-value">{total:,}</div></div>', unsafe_allow_html=True)
with col3:
    melhor_nota = df['NOTA_MEDIA'].max()
    mun_melhor  = df.loc[df['NOTA_MEDIA'].idxmax(), 'Nome_Municipio']
    st.markdown(f'<div class="metric-card"><div class="metric-title">Maior Nota Média</div><div class="metric-value">{melhor_nota:.1f}</div><div class="metric-sub">{mun_melhor}</div></div>', unsafe_allow_html=True)
with col4:
    pior_nota = df['NOTA_MEDIA'].min()
    mun_pior  = df.loc[df['NOTA_MEDIA'].idxmin(), 'Nome_Municipio']
    st.markdown(f'<div class="metric-card"><div class="metric-title">Menor Nota Média</div><div class="metric-value">{pior_nota:.1f}</div><div class="metric-sub">{mun_pior}</div></div>', unsafe_allow_html=True)

# ── DICIONÁRIO DE INDICADORES ─────────────────────────────────────
INDICATORS_DICT = {
    'NOTAS DO ENEM': {
        'Nota Média Geral':  'NOTA_MEDIA',
        'Nota Redação':      'NU_NOTA_REDACAO',
        'Nota Matemática':   'NU_NOTA_MT',
        'Nota Linguagens':   'NU_NOTA_LC',
        'Nota Humanas':      'NU_NOTA_CH',
        'Nota Natureza':     'NU_NOTA_CN',
    },
    'RENDA E DESIGUALDADE (Censo 2010)': {
        'Índice de Gini':                    'GINI',
        'Renda per capita (proxy invertido)': 'RDPC_INV',
    },
    'EDUCAÇÃO (Censo 2010)': {
        'Expectativa de Anos de Estudo (18 anos)':          'E_ANOSESTUDO',
        '% com Ensino Fundamental Completo (25+ anos)':     'PERC_FUND_COMP',
        '% com Ensino Médio Completo (25+ anos)':           'PERC_MED_COMP',
        '% de 15-17 anos ainda no Ensino Fundamental':      'T_FREQ1517_FUND',
        'Taxa de Analfabetismo (15+ anos)':                 'TX_ANALF',
    },
    'DESENVOLVIMENTO HUMANO (Censo 2010)': {
        'IDHM Geral':    'IDHM',
        'IDHM Renda':    'IDHM_R',
        'IDHM Educação': 'IDHM_E',
    },
    'VULNERABILIDADE (Censo 2010)': {
        '% Crianças em Domicílios Sem Ens. Fundamental':    'CRIAN_VULN',
        '% 15-24 anos Nem Estuda Nem Trabalha (vulnerável)': 'NEET_VULN',
    },
    'GASTOS EDUCACIONAIS (SIOPE 2023)': {
        'Investimento por Aluno (R$)':          'EDU_Investimento_Aluno',
        '% de Impostos Aplicados em Educação':  'EDU_Perc_Aplicacao',
        'Total Aplicado em Educação (R$)':      'EDU_Aplicacao_Total',
        'Total de Alunos Matriculados':         'EDU_Alunos',
        'Receita FUNDEB (R$)':                  'EDU_ReceitaFUNDEB',
    },
}

# Filtra apenas colunas presentes no dataframe com dados suficientes
available_metrics  = {}
item_categories    = {}
for cat, items in INDICATORS_DICT.items():
    for label, col in items.items():
        if col in df.columns and df[col].count() > 1:
            display_label = f"[{cat}] {label}"
            available_metrics[display_label] = col
            item_categories[display_label]   = cat

# ── ABAS ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Explorador de Correlação",
    "🌡️ Matriz de Correlação",
    "🎯 Outliers e Distribuição",
])

# ─── TAB 1: EXPLORADOR ────────────────────────────────────────────
with tab1:
    st.markdown("### 📊 Explorador de Correlação")
    st.markdown("Selecione dois indicadores para visualizar a relação entre eles nos 78 municípios.")

    col_var_x, col_var_y = st.columns(2)

    with col_var_x:
        st.markdown("<p style='color:#94A3B8;margin-bottom:0'>Variável Eixo X:</p>", unsafe_allow_html=True)
        cat_x   = st.selectbox("Categoria X:", list(INDICATORS_DICT.keys()), key='cat_x', label_visibility="collapsed")
        opts_x  = {k: v for k, v in INDICATORS_DICT[cat_x].items() if v in df.columns and df[v].count() > 1}
        label_x = st.selectbox("Indicador X:", list(opts_x.keys()), key='ind_x', label_visibility="collapsed")
        x_col   = opts_x[label_x]

    with col_var_y:
        st.markdown("<p style='color:#94A3B8;margin-bottom:0'>Variável Eixo Y:</p>", unsafe_allow_html=True)
        cats_y          = [c for c in INDICATORS_DICT.keys() if c != cat_x]
        default_y_idx   = cats_y.index('NOTAS DO ENEM') if 'NOTAS DO ENEM' in cats_y else 0
        cat_y           = st.selectbox("Categoria Y:", cats_y, index=default_y_idx, key='cat_y', label_visibility="collapsed")

        opts_y = {}
        for label, col in INDICATORS_DICT[cat_y].items():
            if col not in df.columns or df[col].count() <= 1:
                continue
            corr = df[x_col].corr(df[col])
            if pd.isna(corr):
                continue
            if abs(corr) < 0.3:
                display_label = f"🧊 {label} (Fraca)"
            elif corr < -0.3:
                display_label = f"🔻 {label} (Negativa)"
            else:
                display_label = f"🚀 {label} (Positiva)"
            opts_y[display_label] = col

        if not opts_y:
            st.warning("Sem dados correlacionáveis nesta categoria.")
            st.stop()

        label_y    = st.selectbox("Indicador Y:", list(opts_y.keys()), key='ind_y', label_visibility="collapsed")
        y_col      = opts_y[label_y]
        y_axis_label = label_y.split(' (')[0][2:].strip()

    # Correlação
    corr_val = df[x_col].corr(df[y_col])
    if corr_val > 0.6:
        corr_desc = '🟢 Forte e Positiva'
    elif corr_val < -0.6:
        corr_desc = '🔴 Forte e Negativa'
    elif abs(corr_val) > 0.3:
        corr_desc = '🟡 Moderada'
    else:
        corr_desc = '⚪ Fraca / Inexistente'

    st.markdown(f"**Correlação de Pearson:** `r = {corr_val:.3f}` — {corr_desc}")

    fig = px.scatter(
        df, x=x_col, y=y_col,
        hover_name="Nome_Municipio",
        size="QTD_CANDIDATOS",
        color="IDHM" if "IDHM" in df.columns else None,
        color_continuous_scale=px.colors.sequential.Tealgrn,
        trendline="ols",
        labels={x_col: label_x, y_col: y_axis_label},
        title=f"{y_axis_label} × {label_x}",
        template="plotly_dark",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🗃️ Dados por Município")
    display_candidates = [
        'Nome_Municipio', 'QTD_CANDIDATOS', 'NOTA_MEDIA',
        'IDHM', 'IDHM_R', 'IDHM_E',
        'GINI', 'TX_ANALF', 'NEET_VULN', 'CRIAN_VULN',
        'EDU_Investimento_Aluno', 'EDU_Perc_Aplicacao',
    ]
    display_cols = [c for c in display_candidates if c in df.columns]
    st.dataframe(
        df[display_cols].sort_values('NOTA_MEDIA', ascending=False).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

# ─── TAB 2: MATRIZ DE CORRELAÇÃO ──────────────────────────────────
with tab2:
    st.markdown("### 🌡️ Matriz de Correlação entre Indicadores")
    st.markdown("Valores próximos de **+1** indicam correlação positiva forte; próximos de **-1**, negativa forte; próximos de **0**, sem relação.")

    # Permitir que o usuário selecione quais grupos incluir
    grupos_disponiveis = list(INDICATORS_DICT.keys())
    grupos_selecionados = st.multiselect(
        "Grupos de indicadores a incluir na matriz:",
        grupos_disponiveis,
        default=grupos_disponiveis,
        key='grupos_matriz'
    )

    cols_matriz = []
    labels_matriz = {}
    for cat in grupos_selecionados:
        for label, col in INDICATORS_DICT[cat].items():
            if col in df.columns and df[col].count() > 1 and col not in cols_matriz:
                cols_matriz.append(col)
                labels_matriz[col] = label

    if len(cols_matriz) < 2:
        st.warning("Selecione ao menos dois grupos para gerar a matriz.")
    else:
        corr_matrix = df[cols_matriz].corr()
        corr_matrix.index   = [labels_matriz.get(c, c) for c in corr_matrix.index]
        corr_matrix.columns = [labels_matriz.get(c, c) for c in corr_matrix.columns]

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            origin="lower",
            template="plotly_dark",
        )
        fig_corr.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=700,
            font=dict(color="#E2E8F0", size=11),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ─── TAB 3: OUTLIERS ──────────────────────────────────────────────
with tab3:
    st.markdown("### 🎯 Detecção de Outliers e Distribuição")
    st.markdown("Identifique municípios fora da curva em qualquer indicador disponível.")

    # Agora permite qualquer indicador, não só notas
    all_labels = list(available_metrics.keys())
    outlier_label = st.selectbox(
        "Selecione o indicador:",
        all_labels,
        index=0,
        key='outlier1',
    )
    out_col = available_metrics[outlier_label]
    out_display = outlier_label.split('] ')[-1]

    col_box, col_bar = st.columns(2)

    with col_box:
        fig_box = px.box(
            df, y=out_col,
            points="all",
            hover_name="Nome_Municipio",
            title=f"Boxplot — {out_display}",
            template="plotly_dark",
        )
        fig_box.update_traces(marker=dict(size=7, color="#38BDF8"), boxmean=True)
        fig_box.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with col_bar:
        df_sorted   = df.sort_values(out_col, ascending=False).dropna(subset=[out_col])
        df_extremes = pd.concat([df_sorted.head(10), df_sorted.tail(5)]) if len(df_sorted) > 15 else df_sorted

        fig_bar = px.bar(
            df_extremes,
            x="Nome_Municipio",
            y=out_col,
            color="IDHM" if "IDHM" in df.columns else None,
            title=f"Top 10 maiores / 5 menores — {out_display}",
            color_continuous_scale="Viridis",
            template="plotly_dark",
            category_orders={"Nome_Municipio": df_extremes["Nome_Municipio"].tolist()},
        )
        fig_bar.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Tabela de outliers detectados (IQR)
    st.markdown("#### Municípios identificados como outliers (método IQR)")
    q1, q3 = df[out_col].quantile(0.25), df[out_col].quantile(0.75)
    iqr     = q3 - q1
    outliers = df[(df[out_col] < q1 - 1.5 * iqr) | (df[out_col] > q3 + 1.5 * iqr)]
    if outliers.empty:
        st.info("Nenhum outlier detectado para este indicador.")
    else:
        show_cols = ['Nome_Municipio', out_col, 'NOTA_MEDIA', 'IDHM', 'QTD_CANDIDATOS']
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(
            outliers[show_cols].sort_values(out_col, ascending=False).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
