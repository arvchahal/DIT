"""
Universal subscriber for any HuggingFace pipeline task type.
Handles text-generation, text-classification, summarization, etc.

Usage:
  # Text generation
  python sub.py --model-id travel --task text-generation --model-name google/gemma-2b-it

  # Sentiment classification
  python sub.py --model-id personal_finance --task text-classification --model-name distilbert-base-uncased-finetuned-sst-2-english

  # Summarization
  python sub.py --model-id literature --task summarization --model-name google/pegasus-xsum
"""

import os
import sys
import argparse
import asyncio

THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
sys.path.insert(0, SRC_ROOT)

from microservice.subscriber import Subscriber
from dit_components.dit_expert import DitExpert


def maybe_login(hf_token: str | None):
    token = hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import login
        login(token)
        print("[sub] HF login OK.")
    except Exception as e:
        print(f"[sub] HF login warning: {e}")


def build_callable(task: str, hf_pipeline, model_name: str,
                   max_new_tokens: int, temperature: float):
    """Build a str->str callable appropriate for the pipeline task type."""

    if task == "text-generation":
        # Try to load tokenizer for chat templating
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"[sub] tokenizer warning: {e}")

        def text_gen(query: str) -> str:
            prompt = query.strip()
            if tokenizer and hasattr(tokenizer, "apply_chat_template"):
                try:
                    messages = [{"role": "user", "content": prompt}]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt = f"You are a helpful assistant.\n\n{prompt}"

            out = hf_pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),
                top_p=1.0,
                repetition_penalty=1.05,
                truncation=True,
                return_full_text=False,
            )
            if isinstance(out, list) and out:
                if isinstance(out[0], dict) and "generated_text" in out[0]:
                    return out[0]["generated_text"]
                if isinstance(out[0], str):
                    return out[0]
            return str(out)

        return text_gen

    elif task == "text-classification":
        def classify(query: str) -> str:
            out = hf_pipeline(query.strip(), truncation=True)
            if isinstance(out, list) and out and isinstance(out[0], dict):
                label = out[0].get("label", "")
                score = out[0].get("score", 0.0)
                return f"{label} ({score:.4f})"
            return str(out)

        return classify

    elif task == "summarization":
        def summarize(query: str) -> str:
            out = hf_pipeline(query.strip(), max_length=128, min_length=16,
                              truncation=True)
            if isinstance(out, list) and out and isinstance(out[0], dict):
                return out[0].get("summary_text", str(out[0]))
            return str(out)

        return summarize

    else:
        # Generic fallback — just stringify whatever the pipeline returns
        def generic(query: str) -> str:
            out = hf_pipeline(query.strip(), truncation=True)
            if isinstance(out, list) and out:
                if isinstance(out[0], dict):
                    # Return first non-metadata value
                    for key in ("generated_text", "summary_text", "translation_text",
                                "label", "answer"):
                        if key in out[0]:
                            return str(out[0][key])
                    return str(out[0])
                return str(out[0])
            return str(out)

        return generic


async def amain():
    ap = argparse.ArgumentParser(description="Universal DIT Subscriber")
    ap.add_argument("--model-id", required=True,
                    help="Expert id used for NATS subject (e.g., 'travel')")
    ap.add_argument("--nats-url", default="nats://127.0.0.1:4222")
    ap.add_argument("--queue-group", default=None)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--task", required=True,
                    help="HF pipeline task: text-generation, text-classification, summarization, etc.")
    ap.add_argument("--model-name", required=True,
                    help="HF model id (e.g., google/gemma-2b-it)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    maybe_login(args.hf_token)

    # Load model
    expert = DitExpert()
    expert.load_model(task=args.task, model_name=args.model_name)
    hf_pipeline = expert.model

    # Build task-appropriate callable
    expert.model = build_callable(
        args.task, hf_pipeline, args.model_name,
        args.max_new_tokens, args.temperature,
    )

    print(f"[sub] Starting '{args.model_id}' ({args.task}: {args.model_name}) "
          f"on {args.nats_url} ...")
    sub = Subscriber(args.nats_url, args.model_id, expert,
                     queue_group=args.queue_group)
    await sub.run()


if __name__ == "__main__":
    asyncio.run(amain())
