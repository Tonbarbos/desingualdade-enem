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
</style>
""", unsafe_allow_html=True)

def load_data():
    try:
        df = pd.read_csv("dados_compilados.csv")
        return df
    except FileNotFoundError:
        return None

df = load_data()

st.title("🎓 Impacto da Desigualdade Social e Gastos Educacionais nas Notas do ENEM (ES)")
st.markdown("Uma análise exploratória das notas do **ENEM 2024** no Espírito Santo cruzadas com dados socioeconômicos (IBGE/PNAD) e investimentos educacionais municipais (SIOPE).")
st.markdown("---")

if df is None:
    st.error("O arquivo de dados **dados_compilados.csv** não foi encontrado. Execute o script `process_dados.py` primeiro.")
else:
    # 1. Overview Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Municípios Analisados</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total de Candidatos (ES)</div><div class="metric-value">{df["QTD_CANDIDATOS"].sum():,}</div></div>', unsafe_allow_html=True)
    with col3:
        melhor_nota = df['NOTA_MEDIA'].max()
        mun_melhor = df.loc[df['NOTA_MEDIA'].idxmax(), 'Nome_Municipio']
        st.markdown(f'<div class="metric-card"><div class="metric-title">Maior Nota Média</div><div class="metric-value">{melhor_nota:.1f}</div><div style="color: #64748b; font-size: 0.8rem; margin-top:5px;">{mun_melhor}</div></div>', unsafe_allow_html=True)
    with col4:
        pior_nota = df['NOTA_MEDIA'].min()
        mun_pior = df.loc[df['NOTA_MEDIA'].idxmin(), 'Nome_Municipio']
        st.markdown(f'<div class="metric-card"><div class="metric-title">Menor Nota Média</div><div class="metric-value">{pior_nota:.1f}</div><div style="color: #64748b; font-size: 0.8rem; margin-top:5px;">{mun_pior}</div></div>', unsafe_allow_html=True)

    # Criação de Abas (Tabs) para separar Visão Global e Análise de Outliers
    tab1, tab2 = st.tabs(["📊 Visão Global e Correlações", "🎯 Análise de Outliers (Distribuição)"])

    INDICATORS_DICT = {
        'NOTAS DO ENEM': {
            'Nota Média Geral': 'NOTA_MEDIA',
            'Nota Redação': 'NU_NOTA_REDACAO',
            'Nota Matemática': 'NU_NOTA_MT',
            'Nota Linguagens': 'NU_NOTA_LC',
            'Nota Humanas': 'NU_NOTA_CH',
            'Nota Natureza': 'NU_NOTA_CN',
        },
        'RENDA (Censo 2010)': {
            'Renda Per Capita': 'RDPC',
            'Índice de Gini': 'GINI',
        },
        'EDUCAÇÃO (Censo 2010)': {
            'Expectativa de Anos de Estudo (18 anos)': 'E_ANOSESTUDO',
            '% com Ensino Fundamental Completo (25+)': 'PERC_FUND_COMP',
            '% com Ensino Médio Completo (25+)': 'PERC_MED_COMP',
            '% de 15-17 anos na Escola': 'T_FREQ1517',
            'Taxa de Analfabetismo (15+ anos)': 'TX_ANALF',
        },
        'DESENVOLVIMENTO HUMANO (Censo/PNAD)': {
            'IDHM Renda': 'IDHM_R',
            'IDHM Educação': 'IDHM_E',
            'IDHM Geral': 'IDHM',
            'IDHM Renda Ajustado (PNAD)': 'IDHM_R_AJUST',
            'IDHM Ajustado (PNAD)': 'IDHM_AJUST',
        },
        'VULNERABILIDADE (Censo)': {
            '% Crianças em Domic. Sem Ens. Fund.': 'CRIAN_VULN',
            '% 15-24 anos: Nem Estuda/Trabalha (vuln.)': 'NEET_VULN',
        },
        'GASTOS EDUCACIONAIS': {
            'Investimento por Aluno': 'EDU_Investimento_Aluno',
            '% de Impostos Aplicados em Educação': 'EDU_Perc_Aplicacao',
            'Total Aplicado em Educação (R$)': 'EDU_Aplicacao_Total',
            'Total de Alunos Matriculados': 'EDU_Alunos',
        }
    }

    available_metrics = {}
    item_categories = {}
    for cat, items in INDICATORS_DICT.items():
        for label, col in items.items():
            if col in df.columns and df[col].count() > 1:
                display_label = f"[{cat}] {label}"
                available_metrics[display_label] = col
                item_categories[display_label] = cat

    def build_options(exclude_col=None, exclude_category=None):
        opts = []
        for display_label, col in available_metrics.items():
            if exclude_col and col == exclude_col:
                continue
            if exclude_category and item_categories[display_label] == exclude_category:
                continue
            opts.append(display_label)
        return opts

    with tab1:
        # 2. Correlação e Visualização Interativa
        st.markdown("### 📊 Explorador de Correlação")

        col_var_x, col_var_y = st.columns(2)

        with col_var_x:
            st.markdown("<p style='color: #94A3B8; margin-bottom: 0px;'>Variável Eixo X (Causa):</p>", unsafe_allow_html=True)
            
            # Passo 1: Escolher a categoria
            cat_x = st.selectbox("Categoria X:", list(INDICATORS_DICT.keys()), key='cat_x', label_visibility="collapsed")
            
            # Passo 2: Escolher o indicador da categoria
            opts_x_dict = {k: v for k, v in INDICATORS_DICT[cat_x].items() if v in df.columns and df[v].count() > 1}
            label_x = st.selectbox("Indicador X:", list(opts_x_dict.keys()), key='ind_x', label_visibility="collapsed")
            x_col = opts_x_dict[label_x]
            x_axis_label = label_x  # Usado para o título do gráfico

        with col_var_y:
            st.markdown("<p style='color: #94A3B8; margin-bottom: 0px;'>Variável Eixo Y (Efeito):</p>", unsafe_allow_html=True)
            
            # Filtrar categorias para bloquear a mesma categoria escolhida no X (prevenir autocorrelação de temas)
            cats_y = [c for c in INDICATORS_DICT.keys() if c != cat_x]
            
            # Garante que Eixo Y comece selecionando a Categoria "NOTAS DO ENEM" se estiver disponível
            default_cat_y_idx = cats_y.index('NOTAS DO ENEM') if 'NOTAS DO ENEM' in cats_y else 0
            
            # Passo 1: Escolher a categoria
            cat_y = st.selectbox("Categoria Y:", cats_y, index=default_cat_y_idx, key='cat_y', label_visibility="collapsed")
            
            # Passo 2: Escolher o indicador (avaliando a correlação antes para colocar ícones/textos visuais)
            opts_y_dict = {}
            for label, col in INDICATORS_DICT[cat_y].items():
                if col in df.columns and df[col].count() > 1:
                    corr = df[x_col].corr(df[col])
                    if pd.isna(corr):
                        continue
                        
                    # Simulando o "fade out / escurinho" com ícones e avisos textuais diretos na opção, 
                    # já que o st.selectbox nativo não permite estilar HTML (color: rgba) individual
                    if abs(corr) < 0.3:
                        display_label = f"🧊 {label} (Fraca)"
                    elif corr <= -0.3:
                        display_label = f"🔻 {label} (Negativa)"
                    else:
                        display_label = f"🚀 {label} (Positiva)"
                        
                    opts_y_dict[display_label] = col
                    
            if not opts_y_dict:
                st.warning("Sem dados correlacionáveis desta categoria.")
                y_col = x_col
                y_axis_label = "N/A"
            else:
                label_y = st.selectbox("Indicador Y:", list(opts_y_dict.keys()), key='ind_y', label_visibility="collapsed")
                y_col = opts_y_dict[label_y]
                # Limpa a string do label pro gráfico tirando os emojis e textos para o titulo do grafico ficar limpo
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
        st.markdown(f"**Correlação (Pearson):** `{corr_val:.3f}` — {corr_desc}")

        fig = px.scatter(
            df, x=x_col, y=y_col, hover_name="Nome_Municipio", size="QTD_CANDIDATOS",
            color="IDHM" if "IDHM" in df.columns else None, color_continuous_scale=px.colors.sequential.Tealgrn,
            title=f"{y_axis_label} vs {x_axis_label}", template="plotly_dark"
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0"), height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 3. Matriz de Correlação Global
        st.markdown("### 🌡️ Matriz Térmica de Fatores Socioeducacionais")
        st.markdown("Explore como cada variável contribui positiva ou negativamente para o sucesso no ENEM.")
        
        numeric_cols = list(dict.fromkeys(available_metrics.values()))  # únicos, preservando ordem
        numeric_cols = [c for c in numeric_cols if c in df.columns and df[c].count() > 1]
        corr_matrix = df[numeric_cols].corr()

        # Renomeia eixos da matriz removendo o agrupador visual para ficar mais limpo
        col_to_label = {col: label.split('] ')[-1] if '] ' in label else label for label, col in available_metrics.items()}
        corr_matrix.index   = [col_to_label.get(c, c) for c in corr_matrix.index]
        corr_matrix.columns = [col_to_label.get(c, c) for c in corr_matrix.columns]

        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu", origin="lower", template="plotly_dark"
        )
        fig_corr.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=750)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("---")
        
        # 4. Dados Brutos
        st.markdown("### 🗃️ Dados por Município")
        
        # Colunas a exibir — apenas as que existem no dataframe
        display_cols_candidates = [
            'Nome_Municipio', 'QTD_CANDIDATOS', 'NOTA_MEDIA', 'NU_NOTA_REDACAO',
            'GINI', 'RDPC', 'IDHM', 'IDHM_E',
            'EDU_Investimento_Aluno', 'EDU_Alunos',
            'EDU_Perc_Aplicacao', 'EDU_Aplicacao_Total'
        ]
        display_cols = [c for c in display_cols_candidates if c in df.columns]
        
        st.dataframe(
            df[display_cols].sort_values(by='NOTA_MEDIA', ascending=False),
            use_container_width=True, hide_index=True
        )

    with tab2:
        st.markdown("### 🎯 Avaliação de Outliers e Dispersão")
        st.markdown("Utilize o **Diagrama de Caixa (Boxplot)** e o **Gráfico de Barras** para identificar comportamentos anormais, desigualdade extrema ou municípios 'Ponto Fora da Curva' nas métricas estudadas.")
        
        # Filtro de métricas: Deixando apenas as notas do ENEM, conforme solicitado. 
        # As demais métricas socioeducacionais e econômicas ficarão comentadas.
        outlier_metrics_dict = {
            'Nota Média Geral ENEM': 'NOTA_MEDIA',
            'Nota Redação': 'NU_NOTA_REDACAO',
            'Nota Matemática': 'NU_NOTA_MT',
            'Nota Linguagens': 'NU_NOTA_LC',
            'Nota Humanas': 'NU_NOTA_CH',
            'Nota Natureza': 'NU_NOTA_CN',
            # 'Índice de Gini': 'GINI',
            # 'Renda per capita': 'RDPC',
            # '% Extrema Pobreza': 'PMPOB',
            # 'IDHM': 'IDHM',
            # 'IDHM Educação': 'IDHM_E',
            # 'Taxa Frequencia 15-17': 'T_FREQ1517',
            # 'Expectativa Anos Estudo': 'E_ANOSESTUDO',
            # 'Crianças com Domicilio Sem Fund. Comp.': 'CRIAN_VULN',
            # 'Taxa de Desocupação 18-24': 'T_DES1017',
            # 'Gasto Educacional por Aluno (SIOPE)': 'SIOPE_Investimento_Aluno',
            # '% de Aplicação em Educação (SIOPE)': 'SIOPE_Perc_Aplicacao'
        }
        outlier_available_metrics = {k: v for k, v in outlier_metrics_dict.items() if v in df.columns}

        # Código original da seleção de qual métrica usar inteira comentada:
        # outlier_metric_label = st.selectbox("Selecione o indicador para detecção de outliers:", list(available_metrics.keys()), index=0, key='outlier1')
        # out_col = available_metrics[outlier_metric_label]
        
        outlier_metric_label = st.selectbox("Selecione a nota do ENEM para detecção de outliers:", list(outlier_available_metrics.keys()), index=0, key='outlier1')
        out_col = outlier_available_metrics[outlier_metric_label]

        col_box, col_bar = st.columns(2)
        
        with col_box:
            fig_box = px.box(
                df, y=out_col, points="all", hover_name="Nome_Municipio",
                title=f"Boxplot de {outlier_metric_label}", template="plotly_dark"
            )
            fig_box.update_traces(marker=dict(size=8, color="#38BDF8"), boxmean=True)
            fig_box.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0"))
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col_bar:
            # Seleciona os top 10 maiores e 5 menores
            df_sorted = df.sort_values(by=out_col, ascending=False).dropna(subset=[out_col])
            df_extremes = pd.concat([df_sorted.head(10), df_sorted.tail(5)]) if len(df_sorted) > 15 else df_sorted
            
            fig_bar = px.bar(
                df_extremes, x="Nome_Municipio", y=out_col, color="IDHM" if "IDHM" in df.columns else None,
                title=f"Extremos (Top 10 Maiores e Top 5 Menores) - {outlier_metric_label}",
                color_continuous_scale="Viridis", template="plotly_dark",
                category_orders={"Nome_Municipio": df_extremes["Nome_Municipio"].tolist()}
            )
            fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0"), xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
