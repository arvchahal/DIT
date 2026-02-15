# src/experiments/distributed/test/worker_hf.py
import os
import sys
import argparse
import asyncio

# allow: python3 worker_hf.py  (no PYTHONPATH)
THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
sys.path.insert(0, SRC_ROOT)

from microservice.subscriber import Subscriber
from dit_components.dit_expert import DitExpert


def maybe_login(hf_token: str | None):
    """
    Optional: programmatic Hugging Face login for gated models (e.g., Gemma).
    Uses --hf-token or HUGGINGFACE_HUB_TOKEN env if provided.
    Safe to call even if not needed; failures are just logged.
    """
    token = hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import login
        login(token)
        print("[worker] Hugging Face login OK (token provided).")
    except Exception as e:
        print(f"[worker] HF login warning: {e}")


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="Expert id / subject suffix, e.g., Payments")
    ap.add_argument("--nats-url", default="nats://127.0.0.1:4222")
    ap.add_argument("--queue-group", default=None)

    # Model defaults: instruction-tuned small LLM
    ap.add_argument("--task", default="text-generation")
    ap.add_argument("--model-name", default="google/gemma-2b-it",
                    help="HF model id (e.g., google/gemma-2b-it or TinyLlama/TinyLlama-1.1B-Chat-v1.0)")

    # Generation controls
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)

    # Optional HF auth for gated repos
    ap.add_argument("--hf-token", default=None, help="Hugging Face token (or set HUGGINGFACE_HUB_TOKEN)")

    args = ap.parse_args()

    # Login if token provided (needed for gated models like Gemma)
    maybe_login(args.hf_token)

    # Load model via DitExpert
    expert = DitExpert()
    expert.load_model(task=args.task, model_name=args.model_name)

    # Capture original HF pipeline BEFORE wrapping
    hf_pipeline = expert.model

    # Build tokenizer for chat templating if available
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        tokenizer = None
        print(f"[worker] tokenizer load warning: {e} -- proceeding without chat template.")

    def generate_callable(user_prompt: str) -> str:
        """
        Wrap the pipeline to:
        - apply chat template when tokenizer supports it,
        - enforce gen params,
        - normalize output to a plain string.
        """
        prompt_text = user_prompt.strip()

        # Apply chat template if tokenizer supports it (better instruction following)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                # Try system + user messages first (for models like Llama 3)
                messages = [
                    {"role": "user", "content": prompt_text},
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                # Fallback for models that don't support 'system' role (like Gemma)
                print(f"[worker] Chat template issue ({e}); retrying without system role.")
                messages = [
                    {"role": "user", "content": "You are a concise assistant. Answer directly.\n\n" + prompt_text},
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )


        out = hf_pipeline(
            prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=(args.temperature > 0.0),
            top_p=1.0,
            repetition_penalty=1.05,
            truncation=True,
            return_full_text=False,   # only return the completion
        )

        # Normalize various pipeline return shapes into a string
        if isinstance(out, list):
            if out and isinstance(out[0], dict) and "generated_text" in out[0]:
                return out[0]["generated_text"]
            if out and isinstance(out[0], str):
                return out[0]
        return str(out)

    # Swap the expert callable with our wrapper
    expert.model = generate_callable

    # Start subscriber (robust handler already ensures we always respond)
    sub = Subscriber(args.nats_url, args.model_id, expert, queue_group=args.queue_group)
    await sub.run()


if __name__ == "__main__":
    asyncio.run(amain())
