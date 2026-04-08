import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ===============================================================
# CONFIGURAÇÃO DO APP
# ===============================================================
st.set_page_config(page_title="📈 Predição de Vendas Mensais",
    layout="centered")
# Estilo customizado para o botão (verde claro)
st.markdown("""<style>
    div.stButton > button:first-child {
        background-color: #90EE90; /* verde claro */
        color: black;
        border-radius: 8px;
        border: 1px solid #5fbf5f;
        font-weight: 600;}
    div.stButton > button:first-child:hover {
        background-color: #7CCD7C; /* um pouquinho mais escuro no hover */
        color: black;
        border-color: #4aa84a;}</style>""",
    unsafe_allow_html=True)

st.title("📈 MarketingAI - Predição de Vendas Mensais por Loja e Item")
st.markdown("""Este aplicativo utiliza o modelo treinado no projeto para gerar previsões de vendas
mensais por loja, item e período, considerando preço médio, comportamento temporal,
padrões históricos e informações de cluster.""")

# ===============================================================
# DEFININDO DIRETÓRIOS DO PROJETO
# ===============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_PROCESSED_PATH = BASE_DIR / "data" / "processed" / "df_com_cluster_id.parquet"

# ===============================================================
# CARREGAMENTO DOS MODELOS
# ===============================================================
@st.cache_resource
def carregar_modelos():
    preprocess_predicao = joblib.load(ARTIFACTS_DIR / "preprocess_predicao.pkl")
    best_model = joblib.load(ARTIFACTS_DIR / "best_model.pkl")
    preprocess_cluster = joblib.load(ARTIFACTS_DIR / "preprocess_cluster.pkl")
    kmeans_cluster_model = joblib.load(ARTIFACTS_DIR / "kmeans_cluster_model.pkl")
    return preprocess_predicao, best_model, preprocess_cluster, kmeans_cluster_model


# =========================================================
# CARREGAMENTO DOS DADOS
# =========================================================
@st.cache_data
def carregar_df():
    return pd.read_parquet(DATA_PROCESSED_PATH)

preprocess_predicao, best_model, preprocess_cluster, kmeans_cluster_model = carregar_modelos()
df = carregar_df()

# ===============================================================
# SIDEBAR – ENTRADAS DO USUÁRIO
# ===============================================================
st.sidebar.header("Selecione os parâmetros:")

lojas = sorted(df["store_code"].unique())
items = sorted(df["item"].unique())

store = st.sidebar.selectbox("Loja", lojas)
item = st.sidebar.selectbox("Item", items)

# Horizonte: de todo o histórico até 2 anos após o último ano
ano_min = int(df["year"].min())
ano_max_hist = int(df["year"].max())

anos = list(range(ano_min, ano_max_hist + 3))  # histórico + 2 anos futuros

# por padrão, deixa selecionado o último ano histórico
try:
    ano_default_index = anos.index(ano_max_hist)
except ValueError:
    ano_default_index = len(anos) - 1

ano = st.sidebar.selectbox("Ano da previsão", anos, index=ano_default_index)


mes = st.sidebar.selectbox("Mês", list(range(1, 13)))

# Sugere como valor inicial o preço médio histórico do item na loja (se existir)
df_hist = df[(df["store_code"] == store) & (df["item"] == item)]

if not df_hist.empty:
    mean_price_default = float(df_hist["mean_price"].mean())
else:
    # fallback: média global de mean_price
    mean_price_default = float(df["mean_price"].mean())

mean_price_input = st.sidebar.number_input(
    "Preço médio do item (mean_price)",
    min_value=0.0,
    max_value=10000.0,
    step=0.1,
    value=round(mean_price_default, 2))

st.sidebar.caption(
    "💡 Você pode ajustar o preço médio para simular diferentes cenários de preço "
    "e ver como isso impacta a previsão de vendas.")


# ===============================================================
# FUNÇÕES AUXILIARES
# ===============================================================
def calcular_time_features(ano, mes):
    """
    Calcula o time_index para o ano/mês escolhido
    e as variáveis cíclicas month_sin e month_cos.
    """
    # Último ano e mês do histórico
    last_year = int(df["year"].max())
    last_month = int(df["month"].max())

    # Último time_index real
    last_time_index = int(df["time_index"].max())

    # Quantos meses no futuro estamos prevendo
    months_future = (ano - last_year) * 12 + (mes - last_month)

    # Novo índice de tempo
    time_index = last_time_index + months_future

    # Variáveis cíclicas
    month_sin = float(np.sin(2 * np.pi * mes / 12))
    month_cos = float(np.cos(2 * np.pi * mes / 12))

    return time_index, month_sin, month_cos


def prever_cluster(mean_price, time_index, month_sin, month_cos):
    entrada = np.array([[mean_price, time_index, month_sin, month_cos]])
    entrada_scaled = preprocess_cluster.transform(entrada)
    cluster = int(kmeans_cluster_model.predict(entrada_scaled)[0])
    return cluster

# ===============================================================
# BOTÃO – GERAR PREVISÃO
# ===============================================================
if st.sidebar.button("Gerar previsão de vendas"):

    # 1) Calcula features temporais
    time_index, month_sin, month_cos = calcular_time_features(ano, mes)

    # 2) Estima cluster com as mesmas features usadas no treino
    cluster_id = prever_cluster(mean_price_input, time_index, month_sin, month_cos)

    # 3) Calcula médias históricas necessárias para o modelo
    # média por item (em todas as lojas)
    item_mean = df[df["item"] == item]["sales"].mean()
    # média por loja (todos os itens)
    store_mean = df[df["store_code"] == store]["sales"].mean()
    # média por combinação loja + item
    store_item_mean = df[(df["store_code"] == store) & (df["item"] == item)]["sales"].mean()

    # 4) Monta dataframe de entrada completo (todas as colunas usadas no treino)
    df_input = pd.DataFrame({
        "store_code": [store],
        "item": [item],
        "mean_price": [mean_price_input],
        "year": [ano],
        "month": [mes],
        "time_index": [time_index],
        "month_sin": [month_sin],
        "month_cos": [month_cos],
        "cluster_id": [cluster_id],
        "item_mean_sales": [item_mean],
        "store_mean_sales": [store_mean],
        "store_item_mean_sales": [store_item_mean]})

    # 5) Pré-processa com o mesmo pipeline do notebook e faz a previsão
    X_processado = preprocess_predicao.transform(df_input)
    previsao = float(best_model.predict(X_processado)[0])

    # Arredonda para número inteiro (unidades de vendas)
    previsao_int = int(round(previsao))

    # 6) Exibição no app
    st.subheader("📊 Resultado da previsão")
    st.metric("Vendas previstas (sales)", f"{previsao_int}")

    with st.expander("Detalhes do cenário da previsão"):
        st.write(f"**Loja:** `{store}`")
        st.write(f"**Item:** `{item}`")
        st.write(f"**Período:** `{ano}-{mes:02d}`")
        st.write(f"**Preço médio considerado:** `{mean_price_input:.2f}`")
        st.write(f"**Cluster estimado:** `{cluster_id}`")
        st.write(f"**Média histórica do item:** `{item_mean:.2f}`")
        st.write(f"**Média histórica da loja:** `{store_mean:.2f}`")
        st.write(f"**Média histórica loja+item:** `{store_item_mean:.2f}`")
        st.write(f"**Valor contínuo previsto pelo modelo:** {previsao:.2f}")

