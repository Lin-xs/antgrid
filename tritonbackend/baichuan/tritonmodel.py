"""Server for Baichuan13B."""
import base64
import io
import os
import logging
import json

import numpy as np
import torch  # pytype: disable=import-error
from diffusers import StableDiffusionPipeline  # pytype: disable=import-error
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

from pytriton.decorators import batch, first_value, group_by_values

current_file = __file__
absolute_path = os.path.abspath(current_file)
parent_dir = os.path.dirname(absolute_path)

LOGGER = logging.getLogger("examples.huggingface_baichuan_13b_chat.server")
LOGGER.setLevel(logging.DEBUG)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_FORMAT = "JPEG"

dir_name = "Baichuan-13B-Chat"
repo_name = f"{parent_dir}/{dir_name}"
repo_name = "/home/wyq/antgrid_models/Baichuan-13B-Chat"
class PytritonModel():
    def __init__(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            repo_name,
            torch_dtype=torch.float16,
            # device_map='auto',
            trust_remote_code=True)
        self.model = self.model.quantize(8).to(DEVICE)
        self.model.generation_config = GenerationConfig.from_pretrained(
            repo_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            repo_name,
            use_fast=False,
            trust_remote_code=True
        )
        self.iter_dict = {}
    #@group_by_values("img_size")
    #@first_value("img_size")
    @batch
    def _infer_fn(
        self,
        input: np.ndarray,
    ):
        LOGGER.warning("receive infp.")
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
        return {"response": np.char.encode(np.array([[response]])), "history": np.array([[json.dumps(history_ls)]])}

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

        return {"response": np.char.encode(np.array([["NULL"]])), "history": np.array([[json.dumps(history_ls)]])}

