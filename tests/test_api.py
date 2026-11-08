"""API tests with a stand-in classifier (no model download)."""
import pytest
from fastapi.testclient import TestClient

import api


def fake_classifier(label, score):
    return lambda text: [{"label": label, "score": score}]


@pytest.fixture
def client():
    return TestClient(api.app)


def test_root(client):
    assert client.get("/").json()["message"].endswith("is running")


@pytest.mark.parametrize(
    "raw_label,expected",
    [("LABEL_0", "negative"), ("LABEL_1", "positive"), ("SOMETHING_ELSE", "SOMETHING_ELSE")],
)
def test_predict_maps_labels(client, monkeypatch, raw_label, expected):
    monkeypatch.setattr(api, "get_classifier", lambda: fake_classifier(raw_label, 0.9))
    body = client.post("/predict", params={"text": "great movie"}).json()
    assert body["label"] == expected


def test_confidence_is_rounded_to_four_places(client, monkeypatch):
    monkeypatch.setattr(api, "get_classifier", lambda: fake_classifier("LABEL_1", 0.987654321))
    assert client.post("/predict", params={"text": "great"}).json()["confidence"] == 0.9877


def test_text_is_required(client):
    assert client.post("/predict").status_code == 422


def test_model_source_prefers_env_then_local_then_hub(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODEL_ID", raising=False)
    assert api.resolve_model_source() == api.HUB_MODEL_ID

    (tmp_path / "sentiment_model").mkdir()
    assert api.resolve_model_source() == str(api.LOCAL_MODEL_DIR)

    monkeypatch.setenv("MODEL_ID", "someone/other-model")
    assert api.resolve_model_source() == "someone/other-model"
