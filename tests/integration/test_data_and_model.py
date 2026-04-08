import pytest
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_PATH = BASE_DIR / "data" / "processed" / "df_com_cluster_id.parquet"


@pytest.mark.integration
def test_dados_processados_carregam():
    df = pd.read_parquet(DATA_PATH)
    assert not df.empty


@pytest.mark.integration
def test_artefatos_de_predicao_carregam():
    preprocess_predicao = joblib.load(ARTIFACTS_DIR / "preprocess_predicao.pkl")
    best_model = joblib.load(ARTIFACTS_DIR / "best_model.pkl")

    assert preprocess_predicao is not None
    assert best_model is not None


@pytest.mark.integration
def test_colunas_base_existentes():
    df = pd.read_parquet(DATA_PATH)

    colunas_base = {"item","store_code","sales","mean_price","time_index","month_sin", "month_cos",
        "cluster_id"}

    assert colunas_base.issubset(df.columns)


@pytest.mark.integration
def test_modelo_prediz_com_features_historicas_calculadas():
    df = pd.read_parquet(DATA_PATH)

    preprocess_predicao = joblib.load(ARTIFACTS_DIR / "preprocess_predicao.pkl")
    best_model = joblib.load(ARTIFACTS_DIR / "best_model.pkl")

    linha_base = df.iloc[[0]].copy()

    item_val = linha_base["item"].iloc[0]
    store_val = linha_base["store_code"].iloc[0]

    item_mean_sales = df.loc[df["item"] == item_val, "sales"].mean()
    store_mean_sales = df.loc[df["store_code"] == store_val, "sales"].mean()
    store_item_mean_sales = df.loc[
        (df["store_code"] == store_val) & (df["item"] == item_val),
        "sales"
    ].mean()

    entrada = pd.DataFrame([{
        "mean_price": linha_base["mean_price"].iloc[0],
        "time_index": linha_base["time_index"].iloc[0],
        "month_sin": linha_base["month_sin"].iloc[0],
        "month_cos": linha_base["month_cos"].iloc[0],
        "cluster_id": linha_base["cluster_id"].iloc[0],
        "item_mean_sales": item_mean_sales,
        "store_mean_sales": store_mean_sales,
        "store_item_mean_sales": store_item_mean_sales,}])

    entrada_proc = preprocess_predicao.transform(entrada)
    pred = best_model.predict(entrada_proc)[0]

    assert pred is not None
    assert float(pred) >= 0