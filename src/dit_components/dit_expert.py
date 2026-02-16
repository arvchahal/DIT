from typing import Optional, Any
from transformers import pipeline
import torch

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
        quantize: Optional[str] = None,
    ) -> None:
        # path 1 ─ already-loaded object
        if model is not None:
            self.model = model
            return

        # path 2 ─ load from HF hub
        if task is None or model_name is None:
            raise ValueError("Provide either `model` or both `task` and `model_name`.")

        if quantize == "8bit":
            self.model = pipeline(
                task,
                model=model_name,
                device_map="auto",
                model_kwargs={"load_in_8bit": True},
            )
            print(f"[DitExpert] Loaded {model_name} in 8-bit quantized mode.")
        elif quantize == "4bit":
            self.model = pipeline(
                task,
                model=model_name,
                device_map="auto",
                model_kwargs={"load_in_4bit": True},
            )
            print(f"[DitExpert] Loaded {model_name} in 4-bit quantized mode.")
        else:
            self.model = pipeline(
                task,
                model=model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            print(f"[DitExpert] Loaded {model_name} in FP16 (no quantization).")

    def run_model(self, query: str):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model(query)
