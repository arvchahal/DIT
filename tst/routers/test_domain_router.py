import pytest
from routers.domain_router import DomainRouter

@pytest.fixture
def router():
    experts = ["Payments", "Search", "Support"]
    mapping_expert_to_descriptors = {
        "Payments": ["finance"],
        "Search": ["find"],
        "Support": ["help"]
    }
    return DomainRouter(experts=experts, mapping_expert_to_descriptors=mapping_expert_to_descriptors)

def test_route_single_domain_word(router):
    assert router.route("finance") == "Payments"
    assert router.route("find") == "Search"
    assert router.route("help") == "Support"

def test_route_multiple_domain_words(router):
    assert router.route("finance find help") == "Payments"
    assert router.route("find find help") == "Search"

def test_route_no_domain_word(router):
    # Should return first expert
    assert router.route("unknown") == "Payments"

def test_route_tie_breaking(router):
    # Should return first in experts list on tie
    assert router.route("finance help") == "Payments"
