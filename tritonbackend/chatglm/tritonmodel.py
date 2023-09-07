"""Server for ChatGLM 6B."""
import base64
import io
import os
import logging
import json

import numpy as np
import torch  # pytype: disable=import-error
from diffusers import StableDiffusionPipeline  # pytype: disable=import-error
from transformers import AutoTokenizer, AutoModel

from pytriton.decorators import batch, first_value, group_by_values

current_file = __file__
absolute_path = os.path.abspath(current_file)
parent_dir = os.path.dirname(absolute_path)

LOGGER = logging.getLogger("examples.huggingface_chatglm.server")
LOGGER.setLevel(logging.DEBUG)

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
IMAGE_FORMAT = "JPEG"
paradir = '/home/yhc/ChatGLM-6B-model-parameter/'
class PytritonModel():
    def __init__(self) -> None:
        self.model = AutoModel.from_pretrained(paradir, trust_remote_code=True).half()
        self.tokenizer = AutoTokenizer.from_pretrained(paradir, trust_remote_code=True)
        self.model = self.model.to(DEVICE)
        self.model = self.model.eval()
        self.iter_dict = {}

    #@group_by_values("img_size")
    #@first_value("img_size")
    @batch
    def _infer_fn(
        self,
        input: np.ndarray,
        history: np.ndarray,
        max_length: np.int64,
        top_p: np.int64,
        temperature: np.int64,
    ):
        input = [np.char.decode(p.astype("bytes"), "utf-8").item() for p in input]
        input = input[0]

        history_s = history[0][0]
        history_ls = json.loads(history_s)

        max_length = int(max_length[0][0])
        top_p = float(top_p[0][0])
        temperature = float(temperature[0][0])


        LOGGER.debug(f"input: {input}")
        LOGGER.debug(f"History: {history_ls}")
        LOGGER.debug(f"max_length: {max_length}, top_p: {top_p}, temperature: {temperature}")


        outputs = []
        response, history_ls = self.model.chat(self.tokenizer, input, history_ls, max_length=max_length, top_p=top_p,temperature=temperature)

        LOGGER.debug(f"response[0]: {response[:10]}")
        return {"response": np.char.encode(np.array([[response]])), "history": np.array([[json.dumps(history_ls)]])}

    @batch
    def _infer_fn_iter(
            self,
            tid: np.int64
    ):
        tid = int(tid[0][0])
        if self.iter_dict.get(tid) is None:
            raise RuntimeError
        
        try:
            response, history_ls = next(self.iter_dict[tid])
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

        history_s = history[0][0]
        history_ls = json.loads(history_s)

        max_length = int(max_length[0][0])
        top_p = float(top_p[0][0])
        temperature = float(temperature[0][0])

        self.iter_dict[tid] = self.model.stream_chat(self.tokenizer, input, history_ls, max_length=max_length, top_p=top_p,temperature=temperature)

        return {"response": np.char.encode(np.array([["NULL"]])), "history": np.array([[json.dumps(history_ls)]])}

def _encode_image_to_base64(image):
    raw_bytes = io.BytesIO()
    image.save(raw_bytes, IMAGE_FORMAT)
    raw_bytes.seek(0)  # return to the start of the buffer
    return base64.b64encode(raw_bytes.read())
