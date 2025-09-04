from dit_components import DIT, DitExpert
from dit_components.dit_orchestrator import DITOrchestrator
from routers.domain_simplified_router import DomainSimplifiedRouter

# ── Build experts ────────────────────────────────────────────────────────────
experts = {}

exp_a = DitExpert()
exp_a.load_model(
    task="sentiment-analysis",
    model_name="distilbert-base-uncased-finetuned-sst-2-english",
)
experts["sentiment analyzer"] = exp_a

exp_b = DitExpert()
exp_b.load_model(
    task="text-classification",
    model_name="cardiffnlp/tweet-topic-21-multi",
)
experts["text classifier"] = exp_b

# ── Define mapping from expert to descriptor ─────────────────────────────────
mapping_expert_to_descriptors = {
    "sentiment analyzer": ["sentiment", "product", "good"],
    "text classifier": ["topic", "classifier"]
}

# ── Router & DIT façade ──────────────────────────────────────────────────────
router = DomainSimplifiedRouter(experts=list(experts.keys()), mapping_expert_to_descriptors=mapping_expert_to_descriptors)
dit = DIT(experts, router)

# ── Orchestrate queries ──────────────────────────────────────────────────────
orch = DITOrchestrator(dit)
queries = [
    "I love this product!",
    "What's the weather like today?",
    "Can you recommend a good restaurant and many topic for topic -ing?",
    "I'm feeling sad.",
    "Tell me about the latest news.",
    "sentiment analyzer",
    "text classifier topic",
    "topic is my favorite"
]

results = orch.run(queries, total_time=30)
for q, r in zip(queries, results):
    print(f"{q!r:40} -> {r}")
