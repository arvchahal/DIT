# dit_components/dit_expert.py
from transformers import pipeline
from typing import Any, Optional


class DitExpert:
    """
    Thin wrapper around a HF pipeline or any callable model.
    """

    def __init__(self, model: Optional[Any] = None):
        self.model: Optional[Any] = model

    def load_model(
        self,
        *,
        model: Optional[Any] = None,
        task: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        # path 1 ─ already-loaded object
        if model is not None:
            self.model = model
            return

        # path 2 ─ load from HF hub
        if task is None or model_name is None:
            raise ValueError("Provide either `model` or both `task` and `model_name`.")

        try:
            self.model = pipeline(task, model_name)
        except Exception as exc:
            raise RuntimeError(f"Error loading model: {exc}") from exc

    def run_model(self, query: str):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model(query)
