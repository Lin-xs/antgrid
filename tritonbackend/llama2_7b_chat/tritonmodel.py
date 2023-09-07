"""Server for ChatGLM 6B."""
import base64
import io
import os
import logging
import json

import numpy as np
import torch  # pytype: disable=import-error
import transformers

from diffusers import StableDiffusionPipeline  # pytype: disable=import-error

from transformers import AutoTokenizer, AutoModel
from typing import Union


from pytriton.decorators import batch

LOGGER = logging.getLogger("examples.huggingface_llama2-7b-chat.server")
LOGGER.setLevel(logging.DEBUG)


class PytritonModel():
    def __init__(self,
                 repo_name_or_dir: str,
                 device: Union[str, int]) -> None:
        self.pipe = transformers.pipeline(
            "text-generation",
            model=repo_name_or_dir,
            torch_dtype=torch.float16,
            device=device,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(repo_name_or_dir)

    @batch
    def _infer_fn(
        self,
        input: np.ndarray,
        max_length: np.int64
    ):
        input = [np.char.decode(p.astype("bytes"), "utf-8").item() for p in input]
        input = input[0]
        max_length = int(max_length[0][0])

        LOGGER.info(f"Prompts: {input}")
        LOGGER.info(f"max_length: {max_length}")

        sequences = self.pipe(
            input,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=max_length
        )

        response = sequences[0]["generated_text"]
        LOGGER.info(f"response: {response[:10]}")

        return {"response": np.char.encode(np.array([[response]]))}

