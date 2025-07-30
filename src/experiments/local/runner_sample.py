from dit_components.dit_expert import DitExpert
from dit_components.dit import DIT
from routers.simple_router import SimpleRouter
from dit_components.orchestrator import DITOrchestrator

# ── Build experts ────────────────────────────────────────────────────────────
experts = {}

exp_a = DitExpert()
exp_a.load_model(
    task="sentiment-analysis",
    model_name="sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english",
)
experts["sentiment"] = exp_a

exp_b = DitExpert()
exp_b.load_model(task="zero-shot-classification", model_name="facebook/bart-large-mnli")
experts["mnli"] = exp_b

# ── Router & DIT façade ──────────────────────────────────────────────────────
router = SimpleRouter(experts=list(experts.keys()))
dit = DIT(experts, router)

# ── Orchestrate five queries ────────────────────────────────────────────────
orch = DITOrchestrator(dit)
queries = [
    "I love the new MacBook!",
    "Translate 'bonjour' to English.",
    "The market tanked today.",
    "CRISPR will revolutionise medicine.",
    "Meh, could be better.",
]

results = orch.run(queries, total_time=30)
for q, r in zip(queries, results):
    print(f"{q!r:40} → {r}")
