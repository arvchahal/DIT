import pytest
from routers.domain_simplified_router import DomainSimplifiedRouter

@pytest.fixture
def router():
    experts = ["Payments", "Search", "Support"]
    mapping_expert_to_descriptors = {
        "Payments": ["finance"],
        "Search": ["find"],
        "Support": ["help"]
    }
    return DomainSimplifiedRouter(experts=experts, mapping_expert_to_descriptors=mapping_expert_to_descriptors)

def test_route_single_domain_word(router):
    assert router.route("finance") == "Payments"
    assert router.route("find") == "Search"
    assert router.route("help") == "Support"

def test_route_multiple_domain_words(router):
    # Should match the first word only
    assert router.route("finance find help") == "Payments"
    assert router.route("find finance help") == "Search"
    assert router.route("help finance find") == "Support"

def test_route_no_domain_word(router):
    # Should return first expert (fallback)
    assert router.route("unknown") in router.experts

def test_route_first_match_priority(router):
    # Should match the first domain word, not tally
    assert router.route("find help finance") == "Search"
    assert router.route("help find finance") == "Support"
