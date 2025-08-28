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