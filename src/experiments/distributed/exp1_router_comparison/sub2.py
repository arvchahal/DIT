import os
import sys
import argparse
import asyncio

# ── Path setup (so you can run without PYTHONPATH) ────────────────────────────
THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
sys.path.insert(0, SRC_ROOT)

from microservice.subscriber import Subscriber
from dit_components.dit_expert import DitExpert


# ── Optional Hugging Face login ───────────────────────────────────────────────
def maybe_login(hf_token: str | None):
    token = hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import login
        login(token)
        print("[subscriber] ✅ Hugging Face login successful.")
    except Exception as e:
        print(f"[subscriber] ⚠️ HF login warning: {e}")


# ── Main async entrypoint ─────────────────────────────────────────────────────
async def amain():
    ap = argparse.ArgumentParser(description="Distributed Inference Tables Subscriber Node")
    ap.add_argument("--model-id", required=True, help="Expert id (e.g., 'sentiment', 'finance', 'travel')")
    ap.add_argument("--nats-url", default="nats://127.0.0.1:4222", help="NATS server endpoint")
    ap.add_argument("--queue-group", default=None, help="Optional queue group for scaling")
    ap.add_argument("--hf-token", default=None, help="Optional Hugging Face token")

    # Model configuration
    ap.add_argument("--task", default="text-generation")
    ap.add_argument("--model-name", default="google/gemma-2b-it",
                    help="HF model name, e.g., 'google/gemma-2b-it' or 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)

    args = ap.parse_args()

    # Authenticate with Hugging Face if needed
    maybe_login(args.hf_token)

    # ── Load model via DitExpert ─────────────────────────────────────────────
    expert = DitExpert()
    expert.load_model(task=args.task, model_name=args.model_name)
    hf_pipeline = expert.model  # original HF pipeline

    # ── Try to build tokenizer for chat templating ───────────────────────────
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        tokenizer = None
        print(f"[subscriber] tokenizer load warning: {e}")

    # ── Wrap inference to normalize I/O ───────────────────────────────────────
    def generate_callable(user_prompt: str) -> str:
        prompt_text = user_prompt.strip()

        # Apply chat template if available (for chat-tuned models)
        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt_text}]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                print(f"[subscriber] chat template fallback: {e}")
                prompt_text = f"You are a helpful assistant.\n\n{prompt_text}"

        # Run inference
        out = hf_pipeline(
            prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=(args.temperature > 0.0),
            top_p=1.0,
            repetition_penalty=1.05,
            truncation=True,
            return_full_text=False,
        )

        # Normalize pipeline outputs
        if isinstance(out, list):
            if out and isinstance(out[0], dict) and "generated_text" in out[0]:
                return out[0]["generated_text"]
            if out and isinstance(out[0], str):
                return out[0]
        return str(out)

    # Attach wrapped generator to expert
    expert.model = generate_callable

    # ── Start NATS subscriber ────────────────────────────────────────────────
    print(f"[subscriber] Starting expert '{args.model_id}' on {args.nats_url} ...")
    sub = Subscriber(args.nats_url, args.model_id, expert, queue_group=args.queue_group)
    await sub.run()


if __name__ == "__main__":
    asyncio.run(amain())
