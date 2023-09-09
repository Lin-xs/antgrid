"""Server for Baichuan13B."""
import json
import logging
from typing import Union

import numpy as np
import torch
from pytriton.decorators import batch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

LOGGER = logging.getLogger("examples.huggingface_baichuan_13b_chat.model")

class PytritonModel():
    def __init__(self,
                 repo_name_or_dir: str,
                 device: Union[str, int]) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            repo_name_or_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True)
        self.model = self.model.quantize(8).to(device)
        self.model.generation_config = GenerationConfig.from_pretrained(
            repo_name_or_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            repo_name_or_dir,
            use_fast=False,
            trust_remote_code=True
        )
        self.model = self.model.eval()
        self.iter_dict = {}

    @batch
    def _infer_fn(
        self,
        input: np.ndarray,
    ):
        prompts = [np.char.decode(p.astype("bytes"), "utf-8").item() for p in input]
        messages = json.loads(prompts[0])
        LOGGER.debug(f"message: {messages}")

        response = self.model.chat(self.tokenizer, messages)
        LOGGER.debug(f"the response[:10]is : {response[:10]}")

        return {"response": np.char.encode(np.array([[response]]))}

    @batch
    def _infer_fn_iter(
        self,
        tid: np.int64
    ):
        tid = int(tid[0][0])
        if self.iter_dict.get(tid) is None:
            raise RuntimeError

        try:
            response, history_ls = next(self.iter_dict[tid]), []
        except StopIteration:
            response, history_ls = "[STOP]", []
            self.iter_dict.pop(tid)

        LOGGER.debug(f"response[0]: {response[:10]}")
        return {
            "response": np.char.encode(np.array([[response]])),
            "history": np.array([[json.dumps(history_ls)]])
        }

    @batch
    def _add_iter(
            self,
            input: np.ndarray,
            history: np.ndarray,
            max_length: np.int64,
            top_p: np.int64,
            temperature: np.int64,
            tid: np.int64
    ):
        tid = int(tid[0][0])
        if self.iter_dict.get(tid) is not None:
            raise RuntimeError
        input = [np.char.decode(p.astype("bytes"), "utf-8").item() for p in input]
        input = input[0]
        messages = json.loads(input)

        history_s = history[0][0]
        history_ls = json.loads(history_s)

        max_length = int(max_length[0][0])
        top_p = float(top_p[0][0])
        temperature = float(temperature[0][0])

        self.iter_dict[tid] = self.model.chat(self.tokenizer, messages, stream=True)

        return {
            "response": np.char.encode(np.array([["NULL"]])),
            "history": np.array([[json.dumps(history_ls)]])
        }

