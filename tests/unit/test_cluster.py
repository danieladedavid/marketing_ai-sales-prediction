import pytest
from pathlib import Path
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"

def prever_cluster(mean_price, time_index, month_sin, month_cos):
    preprocess_cluster = joblib.load(ARTIFACTS_DIR / "preprocess_cluster.pkl")
    kmeans_cluster_model = joblib.load(ARTIFACTS_DIR / "kmeans_cluster_model.pkl")

    entrada = np.array([[mean_price, time_index, month_sin, month_cos]])
    entrada_scaled = preprocess_cluster.transform(entrada)
    cluster = int(kmeans_cluster_model.predict(entrada_scaled)[0])
    return cluster

@pytest.mark.unit
def test_prever_cluster_retorna_inteiro():
    cluster = prever_cluster(
        mean_price=10.99,
        time_index=6,
        month_sin=0.0,
        month_cos=1.0)
    assert isinstance(cluster, int)

@pytest.mark.unit
def test_prever_cluster_retorna_valor_nao_negativo():
    cluster = prever_cluster(
        mean_price=10.99,
        time_index=6,
        month_sin=0.0,
        month_cos=1.0)
    assert cluster >= 0