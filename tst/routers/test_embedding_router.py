import numpy as np
import pytest
from routers.embedding_router import EmbeddingRouter

@pytest.fixture
def router(monkeypatch):
    monkeypatch.setattr(EmbeddingRouter, "_init_encoder", lambda self: (None, None))

    V = {
        "Payments": np.array([1,0,0], dtype=np.float32),
        "Search":   np.array([0,1,0], dtype=np.float32),
        "Support":  np.array([0,0,1], dtype=np.float32),
        "refund":   np.array([1,0,0], dtype=np.float32),
        "index":    np.array([0,1,0], dtype=np.float32),
        "help":     np.array([0,0,1], dtype=np.float32),
    }

    def fake_get_embedding(self, text: str) -> np.ndarray:
        return V.get(text, V["Support"])

    monkeypatch.setattr(EmbeddingRouter, "_get_embedding", fake_get_embedding, raising=False)

    return EmbeddingRouter(experts=["Payments", "Search", "Support"])

def test_anchor_cache_initialized(router):
    assert list(router.agent_anchors_MRU_cache) == ["Payments", "Search", "Support"]
    assert set(router.anchor_embeddings) == {"Payments", "Search", "Support"}

def test_route_selects_most_similar_and_updates_mru(router):
    assert router.route("refund") == "Payments"
    assert list(router.agent_anchors_MRU_cache)[0] == "Payments"
    assert router.route("index") == "Search"
    assert list(router.agent_anchors_MRU_cache)[0] == "Search"

def test_route_empty_query_cycles(router):
    a = router.route("")
    b = router.route("")
    c = router.route("")
    assert [a,b,c] == ["Payments", "Search", "Support"]

def test_ties_break_by_order(router):
    same = np.array([1,0,0], dtype=np.float32)
    for k in router.anchor_embeddings:
        router.anchor_embeddings[k] = same
    assert router.route("help") == "Payments"  # leftmost wins on tie

def test_encoder_embedding_shape(): 
    router = EmbeddingRouter(experts=["Payments", "Search", "Support"])
    embedding = router._get_embedding("test query")
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.shape[0] == 384

def test_actual_encoder_similarity():
    router = EmbeddingRouter(experts=["Payments", "Search", "Support"])
    embedding1 = router._get_embedding("test query 1")
    embedding2 = router._get_embedding("test query 1")
    similarity = router._cosine_sim(embedding1, embedding2)
    assert np.isclose(similarity, 1.0, atol=1e-3)
