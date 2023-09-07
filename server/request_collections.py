import base64
import io
import numpy as np
from PIL import Image  # pytype: disable=import-error

from pytriton.client import ModelClient
import json


class ChatGLMCall:
    @staticmethod
    def infer(payload):
        input = payload["input"]
        history = payload["history"]
        max_length = payload["max_length"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]

        input = np.char.encode(np.array([[input]]), "utf-8")
        history = np.char.encode(np.array([[json.dumps(history)]]), "utf-8")
        max_length = np.array([[max_length]], dtype=np.int64)
        top_p = np.array([[top_p]], dtype=np.float32)
        temperature = np.array([[temperature]], dtype=np.float32)

        with ModelClient("grpc://localhost:8001", "ChatGLM_6B", init_timeout_s=1200.0) as client:
            result_dict = client.infer_batch(input=input, history=history, max_length=max_length, top_p=top_p, temperature=temperature)
            return result_dict["response"], result_dict["history"]

    @staticmethod
    def infer_iter(tid, method="grpc", address="localhost", port="8001"):
        tid = np.array([[tid]], dtype=np.int64)
        history = []
        response = None
        while True:
            with ModelClient(f"{method}://{address}:{port}", "ChatGLM_6B_fn_iter", init_timeout_s=1200.0) as client:
                result_dict = client.infer_batch(tid=tid)
                response = result_dict["response"][0][0].decode()
                history_tmp = json.loads(result_dict["history"][0][0])
                # print(response)

                if response == "[STOP]":
                    break
                else:
                    history = history_tmp
                    yield response, history
        # 最后一返回：stop，以及记录的所有历史。
        yield response, history

    @staticmethod
    def add_iter(payload, tid, method="grpc", address="localhost", port="8001"):
        input = payload["input"]
        history = payload["history"]
        max_length = payload["max_length"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]

        input = np.char.encode(np.array([[input]]), "utf-8")
        history = np.char.encode(np.array([[json.dumps(history)]]), "utf-8")
        max_length = np.array([[max_length]], dtype=np.int64)
        top_p = np.array([[top_p]], dtype=np.float32)
        temperature = np.array([[temperature]], dtype=np.float32)

        tid = np.array([[tid]], dtype=np.int64)

        with ModelClient(f"{method}://{address}:{port}", "ChatGLM_6B_additer", init_timeout_s=1200.0) as client:
            result_dict = client.infer_batch(input=input, history=history, max_length=max_length, top_p=top_p, temperature=temperature, tid=tid)
            return result_dict["response"], result_dict["history"]


class BaichuanCall:
    @staticmethod
    def infer(payload):
        input = payload["input"]
        history = payload["history"]


        input = np.char.encode(np.array([[input]]), "utf-8")
        history = np.char.encode(np.array([[json.dumps(history)]]), "utf-8")
        max_length = np.array([[2048]], dtype=np.int64)
        top_p = np.array([[0.7]], dtype=np.float32)
        temperature = np.array([[0.9]], dtype=np.float32)

        with ModelClient("grpc://localhost:8001", "Baichuan_13B", init_timeout_s=1200.0) as client:
            result_dict = client.infer_batch(input=input, history=history, max_length=max_length, top_p=top_p, temperature=temperature)
            return result_dict["response"], result_dict["history"]

    @staticmethod
    def infer_iter(tid, method="grpc", address="localhost", port="8001"):
        tid = np.array([[tid]], dtype=np.int64)
        history = []
        response = None
        while True:
            with ModelClient(f"{method}://{address}:{port}", "Baichuan_13B_fn_iter", init_timeout_s=1200.0) as client:
                result_dict = client.infer_batch(tid=tid)
                response = result_dict["response"][0][0].decode()
                history_tmp = json.loads(result_dict["history"][0][0])
                # print(response)

                if response == "[STOP]":
                    break
                else:
                    history = history_tmp
                    yield response, history
        # 最后一返回：stop，以及记录的所有历史。
        yield response, history

    @staticmethod
    def add_iter(payload, tid, method="grpc", address="localhost", port="8001"):
        input = json.dumps(payload["input"])
        history = payload["history"]
        # max_length = payload["max_length"]
        # top_p = payload["top_p"]
        # temperature = payload["temperature"]
        max_length = 2048
        top_p = 0.7
        temperature = 0.9

        input = np.char.encode(np.array([[input]]), "utf-8")
        history = np.char.encode(np.array([[json.dumps(history)]]), "utf-8")
        max_length = np.array([[max_length]], dtype=np.int64)
        top_p = np.array([[top_p]], dtype=np.float32)
        temperature = np.array([[temperature]], dtype=np.float32)

        tid = np.array([[tid]], dtype=np.int64)

        with ModelClient(f"{method}://{address}:{port}", "Baichuan_13B_additer", init_timeout_s=1200.0) as client:
            result_dict = client.infer_batch(input=input, history=history, max_length=max_length, top_p=top_p, temperature=temperature, tid=tid)
            return result_dict["response"], result_dict["history"]

class StableDiffusion_1_5Call:
    @staticmethod
    def infer(payload, tid, method="grpc", address="localhost", port="6001"):
        lora_tag = int(payload["lora_tag"])
        scale = float(payload["scale"])
        prompt = payload["prompt"]
        img_height = int(payload["height"])
        img_width = int(payload["width"])
        inference_steps = int(payload["inference_steps"])

        lora_tag = np.array([[lora_tag]])
        scale = np.array([[scale]], dtype=np.float32)
        prompt = np.array([[prompt]])
        prompt = np.char.encode(prompt, "utf-8")
        img_height = np.array([[img_height]])
        img_width = np.array([[img_width]])
        inference_steps = np.array([[inference_steps]])

        with ModelClient(f"{method}://{address}:{port}", "StableDiffusion_1_5", init_timeout_s=1200.0) as client:
            # result_dict = client.infer_batch(prompt=prompt, img_size=img_size)
            # result_dict = client.infer_batch(prompt=prompt, img_size=img_size, inference_steps=inference_steps)
            result_dict = client.infer_batch(lora_tag = lora_tag,
                                             scale = scale,
                                             prompt=prompt,
                                             img_H = img_height,
                                             img_W = img_width,
                                             num_inference_steps = inference_steps)
            return result_dict["image"][0]

class Llama2Call:
    @staticmethod
    def infer(payload, tid, method="grpc", address="localhost", port="8001"):
        input = payload["input"]
        max_length = payload["max_length"]

        input = np.char.encode(np.array([[input]]), "utf-8")
        max_length = np.array([[max_length]], dtype=np.int64)

        with ModelClient(f"{method}://{address}:{port}", "Llama2-7b-chat", init_timeout_s=1200.0) as client:
            result_dict = client.infer_batch(input=input, max_length=max_length)
            response = result_dict["response"][0][0].decode()
            return response



__all__ = ["ChatGLMCall", "BaichuanCall", "StableDiffusion_1_5Call", "Llama2Call"]

