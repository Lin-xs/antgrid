"""Model for Stable Diffusion 1.5."""
import base64
import io
import logging
from typing import Union

import numpy as np
import torch  # pytype: disable=import-error
from diffusers import StableDiffusionPipeline  # pytype: disable=import-error
from lora_diffusion import patch_pipe, tune_lora_scale
from pytriton.decorators import batch

LOGGER = logging.getLogger("examples.huggingface_stable_diffusion2.1.model")

IMAGE_FORMAT = "JPEG"


class PytritonModel():
    safetensors_list = [
        "lora_disney",
        "lora_popart",
        "lora_krk_inpainting",
        "modern_disney_svd",
        "analog_svd_rank4"
    ]

    def __init__(self,
                 repo_name_or_dir: str,
                 lora_safetensor_dir: str,
                 device: Union[str, int]) -> None:
        self.pipe = StableDiffusionPipeline.from_pretrained(
            repo_name_or_dir,
            torch_dtype=torch.float16
        ).to(device)
        self.lora_state = (0, 0.0) #lora tag & lora scale
        # self.lora_safetensor_dir = lora_safetensor_dir
        # patch_pipe(self.pipe, f"{lora_safetensor_dir}/{self.safetensors_list[0]}.safetensors")
        # tune_lora_scale(self.pipe.unet, 0.0)
        # tune_lora_scale(self.pipe.text_encoder, 0.0)


    def _patch_and_tune(self, lora_tag: int, scale: float):
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

        patch_pipe(self.pipe, f"{self.lora_safetensor_dir}/{self.safetensors_list[lora_tag]}.safetensors")
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

        LOGGER.debug(f"Prompts: {prompts}")
        LOGGER.info(f"Image Size: {img_H}x{img_W}")

        # lora_tag = int(lora_tag[0][0])
        # scale = float(scale[0][0])
        # self.patch_and_tune(lora_tag, scale)

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
            LOGGER.debug(f"Generated result for prompt `{prompts[idx]}` with size {len(raw_data)}")

        LOGGER.debug(f"Prepared batch response of size: {len(outputs)}")
        return {"image": np.array(outputs)}