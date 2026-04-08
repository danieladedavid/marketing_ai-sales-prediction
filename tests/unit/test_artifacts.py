import pytest
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"

@pytest.mark.unit
def test_arquivos_principais_existem():
    assert (ARTIFACTS_DIR / "best_model.pkl").exists()
    assert (ARTIFACTS_DIR / "kmeans_cluster_model.pkl").exists()
    assert (ARTIFACTS_DIR / "preprocess_cluster.pkl").exists()
    assert (ARTIFACTS_DIR / "preprocess_predicao.pkl").exists()

@pytest.mark.unit
def test_artefatos_carregam_sem_erro():
    best_model = joblib.load(ARTIFACTS_DIR / "best_model.pkl")
    kmeans_cluster_model = joblib.load(ARTIFACTS_DIR / "kmeans_cluster_model.pkl")
    preprocess_cluster = joblib.load(ARTIFACTS_DIR / "preprocess_cluster.pkl")
    preprocess_predicao = joblib.load(ARTIFACTS_DIR / "preprocess_predicao.pkl")

    assert best_model is not None
    assert kmeans_cluster_model is not None
    assert preprocess_cluster is not None
    assert preprocess_predicao is not None