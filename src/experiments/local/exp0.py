import os
import csv
import time
from datetime import datetime

from routers.simple_router import SimpleRouter
from routers.domain_router import DomainRouter
from routers.domain_simplified_router import DomainSimplifiedRouter
from routers.embedding_router import EmbeddingRouter


# Run: python src/experiments/local/exp0.py


# Dummy queries
queries = [f"query_{i}" for i in range(100)]

# Dummy expert setup
experts = ["Payments", "Search", "Support"]
mapping_expert_to_descriptors = {
    "Payments": ["finance"],
    "Search": ["find"],
    "Support": ["help"]
}

router_types = {
    "basic": lambda: SimpleRouter(experts=experts),
    "domain": lambda: DomainRouter(experts=experts, mapping_expert_to_descriptors=mapping_expert_to_descriptors),
    "domain_simplified": lambda: DomainSimplifiedRouter(experts=experts, mapping_expert_to_descriptors=mapping_expert_to_descriptors),
    "embedding": lambda: EmbeddingRouter(experts=experts)
}

timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
out_dir = os.path.join("data", "exp0", timestamp)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "results.csv")

with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "router_type", "input", "routing_time"])
    idx = 0
    for router_name, router_factory in router_types.items():
        router = router_factory()
        for q in queries:
            start = time.perf_counter()
            router.route(q)
            routing_time = time.perf_counter() - start
            writer.writerow([idx, router_name, q, routing_time])
            idx += 1

print(f"Results written to {out_path}")