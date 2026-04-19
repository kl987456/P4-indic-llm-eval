"""
HuggingFace local model adapter.
Supports any causal-LM or seq2seq model loadable via transformers.
Includes 4-bit / 8-bit quantization via bitsandbytes.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from .base import ModelAdapter

logger = logging.getLogger(__name__)


class HuggingFaceAdapter(ModelAdapter):
    """
    Load a model locally via HuggingFace transformers.

    Usage::

        adapter = HuggingFaceAdapter(
            "ai4bharat/indic-gemma-7b-instruct",
            device="cuda",
            load_in_4bit=True,
        )
        response = adapter.complete("मुझे भारत के बारे में बताइए।")
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        hf_token: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(model_name, **kwargs)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "transformers / torch not installed. Run: pip install transformers torch"
            ) from exc

        token = hf_token or os.environ.get("HF_TOKEN")

        quant_config = None
        if load_in_4bit or load_in_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=torch.float16,
            )

        logger.info("Loading tokenizer for %s", model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            token=token,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info("Loading model %s (device=%s, 4bit=%s, 8bit=%s)", model_name, device, load_in_4bit, load_in_8bit)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            token=token,
        )
        self._model.eval()

        # Try to use the chat template if available
        self._has_chat_template = hasattr(self._tokenizer, "apply_chat_template")

    # ------------------------------------------------------------------

    def _format_prompt(self, prompt: str) -> str:
        if self._has_chat_template:
            messages = [{"role": "user", "content": prompt}]
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> str:
        import torch

        formatted = self._format_prompt(prompt)
        inputs = self._tokenizer(formatted, return_tensors="pt").to(self._model.device)

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Honour stop sequences
        if stop:
            for s in stop:
                if s in text:
                    text = text[: text.index(s)]

        return text.strip()

    def complete_n(
        self,
        prompt: str,
        n: int,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> List[str]:
        import torch

        formatted = self._format_prompt(prompt)
        inputs = self._tokenizer(formatted, return_tensors="pt").to(self._model.device)
        # Expand batch for n generations
        input_ids = inputs["input_ids"].expand(n, -1)
        attention_mask = inputs["attention_mask"].expand(n, -1)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        with torch.no_grad():
            output_ids = self._model.generate(**gen_kwargs)

        in_len = inputs["input_ids"].shape[-1]
        results = []
        for row in output_ids:
            text = self._tokenizer.decode(row[in_len:], skip_special_tokens=True)
            results.append(text.strip())
        return results
