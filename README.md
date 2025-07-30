# Distributed Inference Tables (DIT)

A lightweight, modular framework for routing inference requests to specialized “expert” models across a distributed network of nodes. Inspired by Distributed Hash Tables (DHTs), DIT lets you:

- **Partition inference**: each node hosts one or more small expert models (e.g. domain‑ or task‑specific).
- **Route dynamically**: inputs are sent to the best expert(s) via configurable routing strategies.
- **Scale horizontally**: add or remove experts at runtime without retraining a monolithic model.
- **Deploy anywhere**: run on edge devices, cloud servers, or a hybrid mix.


## Installation

```bash
git clone https://github.com/your-org/dit.git
cd dit
pip install -r requirements.txt
