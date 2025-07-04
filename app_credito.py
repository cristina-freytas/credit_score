import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import metrics
from scipy.stats import ks_2samp
import plotly.express as px

# Configurações gerais
st.set_page_config(layout='wide')
st.title("Análise de Crédito - Score e Risco")

# Carregamento de dados
@st.cache_data
def carregar_dados():
    df = pd.read_feather('credit_scoring.ftr')
    df['data_ref'] = pd.to_datetime(df['data_ref'])
    return df

df = carregar_dados()

# Últimos 3 meses
ultima_data = df['data_ref'].max()
data_limite = (ultima_data.replace(day=1) - pd.DateOffset(months=3))
df = df[df['data_ref'] >= data_limite].copy()
df.drop(columns=['index'], errors='ignore', inplace=True)
df['mau'] = df['mau'].astype(int)

st.subheader("1. Informações Gerais")
col1, col2 = st.columns(2)
with col1:
    st.write("Número de linhas:", len(df))
    st.write("Distribuição da variável resposta (mau):")
    st.dataframe(df['mau'].value_counts(normalize=True).rename('Proporção (%)') * 100)
with col2:
    st.write("Colunas e tipos:")
    st.dataframe(pd.DataFrame({
        "Tipo": df.dtypes,
        "Nulos": df.isna().sum(),
        "Valores únicos": df.nunique()
    }))

# Análises Gráficas
st.subheader("2. Correlação entre Variáveis Numéricas")
num_cols = df.select_dtypes(include='number').columns
fig_corr, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig_corr)

st.subheader("3. Boxplots: Variáveis Numéricas por Categóricas")
cat_cols = df.select_dtypes(include='object').columns
selected_cat = st.selectbox("Escolha a variável categórica:", cat_cols)
selected_num = st.selectbox("Escolha a variável numérica:", num_cols)

fig_box, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x=selected_cat, y=selected_num, data=df, ax=ax)
ax.set_title(f'{selected_num} por {selected_cat}')
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig_box)

# IV (Weight of Evidence)
def IV(variavel, resposta):
    tab = pd.crosstab(variavel, resposta, margins=True)
    tab['pct_evento'] = tab[1]/tab.loc['All',1]
    tab['pct_nao_evento'] = tab[0]/tab.loc['All',0]
    tab['woe'] = np.log((tab['pct_evento']+1e-6)/(tab['pct_nao_evento']+1e-6))
    tab['iv_parcial'] = (tab['pct_evento'] - tab['pct_nao_evento']) * tab['woe']
    return tab['iv_parcial'].sum()

st.subheader("4. Information Value (IV)")
variavel_iv = st.selectbox("Escolha a variável para cálculo de IV:", df.columns)
try:
    if df[variavel_iv].nunique() > 6:
        var_discretizada = pd.qcut(df[variavel_iv], 5, duplicates='drop')
    else:
        var_discretizada = df[variavel_iv]
    iv_val = IV(var_discretizada, df['mau'])
    st.write(f"IV da variável `{variavel_iv}`: **{iv_val:.2%}**")
except Exception as e:
    st.error(f"Erro ao calcular IV: {e}")

# Modelo de regressão
st.subheader("5. Modelo Logístico")

formula = 'mau ~ sexo + posse_de_imovel + idade + tempo_emprego + tipo_residencia + educacao'
rl = smf.glm(formula, data=df, family=sm.families.Binomial()).fit()
df['score'] = rl.predict(df)

st.write("Resumo do modelo:")
st.text(rl.summary())

# Métricas
st.subheader("6. Métricas do Score")

df_model = df[['mau', 'score']].dropna()
acc = metrics.accuracy_score(df_model['mau'], df_model['score'] > 0.068)
fpr, tpr, _ = metrics.roc_curve(df_model['mau'], df_model['score'])
auc = metrics.auc(fpr, tpr)
gini = 2 * auc - 1
ks = ks_2samp(df_model[df_model.mau == 1]['score'], df_model[df_model.mau == 0]['score']).statistic

st.write(f"Acurácia: **{acc:.1%}**")
st.write(f"AUC: **{auc:.1%}**")
st.write(f"GINI: **{gini:.1%}**")
st.write(f"KS: **{ks:.1%}**")

# Gráficos de perfil
st.subheader("7. Perfil do Score")

df_sorted = df.sort_values(by='score').reset_index()
df_sorted['tx_mau_acum'] = df_sorted.mau.cumsum()/df_sorted.shape[0]
df_sorted['pct_aprovacao'] = np.arange(df_sorted.shape[0]) / df_sorted.shape[0]
df_sorted['pct_mau_acum'] = df_sorted.mau.cumsum()/df_sorted.mau.sum()
df_sorted['red_mau_acum'] = 1 - df_sorted['pct_mau_acum']

fig1 = px.line(df_sorted, x="pct_aprovacao", y="tx_mau_acum", title='Taxa de maus por % aprovação')
fig2 = px.line(df_sorted, x="pct_aprovacao", y="red_mau_acum", title='Redução de inadimplência por % aprovação')
st.plotly_chart(fig1)
st.plotly_chart(fig2)

st.write("App desenvolvido com Streamlit para análise de crédito baseada em score logístico.")
