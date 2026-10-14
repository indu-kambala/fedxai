from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.fallback import grounded_fallback
from llm.prompts import SYSTEM_PROMPT, build_user_prompt

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

_tokenizer = None
_model = None


def load_local_model():
    global _tokenizer, _model

    if _model is not None:
        return _tokenizer, _model

    print(f"Loading local Hugging Face model: {MODEL_NAME}")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    _model.eval()

    print("Local Hugging Face model loaded.")

    return _tokenizer, _model


def generate(
    question: str,
    context: dict[str, Any] | None = None
) -> dict[str, Any]:

    context = context or {}

    fallback_text = grounded_fallback({
        "prediction": context.get("prediction")
    })

    try:
        tokenizer, model = load_local_model()

        user_prompt = build_user_prompt(
            question,
            context
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[
            0,
            inputs["input_ids"].shape[1]:
        ]

        text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        if not text:
            return {
                "content": fallback_text,
                "provider": "fallback",
                "model": MODEL_NAME,
            }

        disclaimer = (
            "\n\nAI-assisted research prototype. "
            "This is clinical decision support, not a diagnosis."
        )

        return {
            "content": text + disclaimer,
            "provider": "huggingface-local",
            "model": MODEL_NAME,
        }

    except Exception as exc:
        print(
            f"Local Hugging Face inference failed: "
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "content": fallback_text
            + " Local Hugging Face inference was unavailable; "
            "using the grounded fallback.",
            "provider": "fallback",
            "model": MODEL_NAME,
        }