"""Model for Stable Diffusion 1.5."""
import argparse
import base64
import io
import logging

import numpy as np
import torch  # pytype: disable=import-error
from diffusers import StableDiffusionPipeline  # pytype: disable=import-error

from pytriton.decorators import batch, first_value, group_by_values
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from lora_diffusion import patch_pipe, tune_lora_scale

LOGGER = logging.getLogger("examples.huggingface_stable_diffusion.server")

DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"
IMAGE_FORMAT = "JPEG"



safetensors_list = [
    "lora_disney",
    "lora_popart",
    "lora_krk_inpainting",
    "modern_disney_svd",
    "analog_svd_rank4"
]

class PytritonModel():
    def __init__(self, lora_safetensor_dir: str) -> None:
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(DEVICE)
        self.lora_state = (0, 0.0) #lora tag & lora scale
        self.lora_safetensor_dir = lora_safetensor_dir
        patch_pipe(self.pipe, f"{lora_safetensor_dir}/{safetensors_list[0]}.safetensors")
        tune_lora_scale(self.pipe.unet, 0.0)
        tune_lora_scale(self.pipe.text_encoder, 0.0)
        print("..............")


    def patch_and_tune(self, lora_tag: int, scale: float):
        tag_now, scale_now = self.lora_state
        if tag_now == lora_tag and scale_now == scale:
            return
        if tag_now == lora_tag:
            tune_lora_scale(self.pipe.unet, scale)
            tune_lora_scale(self.pipe.text_encoder, scale)
            print(type(self.lora_state))
            self.lora_state = (lora_tag, scale)
            LOGGER.info(f"change tag {tag_now}, scale{scale_now} to tag {self.lora_state[0]}, scale{self.lora_state[1]}")
            return

        patch_pipe(self.pipe, f"{self.lora_safetensor_dir}/{safetensors_list[lora_tag]}.safetensors")
        tune_lora_scale(self.pipe.unet, scale)
        tune_lora_scale(self.pipe.text_encoder, scale)
        self.lora_state = (lora_tag, scale)
        LOGGER.info(f"change tag {tag_now}, scale{scale_now} to tag {self.lora_state[0]}, scale{self.lora_state[1]}")

    @staticmethod
    def _encode_image_to_base64(image):
        raw_bytes = io.BytesIO()
        image.save(raw_bytes, IMAGE_FORMAT)
        raw_bytes.seek(0)  # return to the start of the buffer
        return base64.b64encode(raw_bytes.read())

    @batch
    def _infer_fn(
        self,
        lora_tag: np.int64,
        scale: np.float32,
        prompt: np.ndarray,
        img_H: np.int64,
        img_W: np.int64,
        num_inference_steps: np.int64,
    ):
        prompts = [np.char.decode(p.astype("bytes"), "utf-8").item() for p in prompt]

        img_H = int(img_H[0][0])
        img_W = int(img_W[0][0])
        num_inference_steps = int(num_inference_steps[0][0])

        LOGGER.info(f"Prompts: {prompts}")
        LOGGER.info(f"Image Size: {img_H}x{img_W}")

        lora_tag = int(lora_tag[0][0])
        scale = float(scale[0][0])
        self.patch_and_tune(lora_tag, scale)

        outputs = []
        for idx, image in enumerate(
            self.pipe(
                prompt=prompts,
                height=img_H,
                width=img_W,
                num_inference_steps=num_inference_steps
            ).images
        ):
            raw_data = self._encode_image_to_base64(image)
            outputs.append(raw_data)
            LOGGER.info(f"Generated result for prompt `{prompts[idx]}` with size {len(raw_data)}")

        LOGGER.debug(f"Prepared batch response of size: {len(outputs)}")
        return {"image": np.array(outputs)}