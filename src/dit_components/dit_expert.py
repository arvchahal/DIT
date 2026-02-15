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
    ) -> None:
        # path 1 ─ already-loaded object
        if model is not None:
            self.model = model
            return

        # path 2 ─ load from HF hub
        if task is None or model_name is None:
            raise ValueError("Provide either `model` or both `task` and `model_name`.")

        # Try 4-bit quantized first, then 8-bit, then fallback full precision
        try:
            self.model = pipeline(
                    task,
                    model=model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            print(f"[DitExpert] Loaded {model_name} in FP16 (no quantization).")
        #     self.model = pipeline(
        #         task,
        #         model=model_name,
        #         device_map="auto",             # automatically use GPU/CPU
        #         torch_dtype=torch.float16,
        #         model_kwargs={
        #         },
        #     )
        #     print(f"[DitExpert] Loaded {model_name} in 4-bit quantized mode.")
        except Exception as e4:
            print(f"[DitExpert] 4-bit load ({e4}); trying 8-bit.")
        #     try:
        #         self.model = pipeline(
        #             task,
        #             model=model_name,
        #             device_map="auto",
        #             torch_dtype=torch.float16,                )
        #     except Exception as e8:
        #         print(f"[DitExpert] 8-bit load failed ({e8}); falling back to FP16.")
        #         self.model = pipeline(
        #             task,
        #             model=model_name,
        #             device_map="auto",
        #             torch_dtype=torch.float16,
        #         )
        #         print(f"[DitExpert] Loaded {model_name} in FP16 (no quantization).")

    def run_model(self, query: str):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model(query)
