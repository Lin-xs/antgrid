import argparse
import logging
import os

import numpy as np
from pytriton.decorators import batch, first_value, group_by_values
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

import tritonmodel

LOGGER = logging.getLogger("examples.huggingface_stable_diffusion2.1.server")

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging in debug mode.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="/home/yhc/stable-diffusion-2-1",
        help="The directory of huggingface runwayml/stable-diffusion-v1-5."
    )
    parser.add_argument(
        "--lora_dir",
        default="/root/antgrid/tritonbackend/sd1_5/lora_safetensors",
        help="The directory of lora safetensors."
    )
    parser.add_argument(
        "--http_port",
        "--port",
        "-p",
        type=int,
        default=8000,
        help="Triton server's HTTP port."
    )
    parser.add_argument(
        "--grpc_port",
        "--gport",
        type=int,
        help="Triton server's gRPC port. If you set up this port, you should set up metrics port together. Default: HTTP port + 1."
    )
    parser.add_argument(
        "--metrics_port",
        "--mport",
        type=int,
        help="Triton server's metrics port. If you set up this port, you should set up gRPC port together. Default: HTTP port + 1."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="The device you want to run models on. the defalut value is cuda:0 if you have an NVIDIA GPU. Otherwise it is cpu."
    )
    return parser.parse_args()

def main():
    """Initialize server with model."""
    args = _parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    LOGGER.info(f"repo dir: {args.repo}")
    log_verbose = 1 if args.verbose else 0

    http_port = args.http_port
    if args.grpc_port is None and args.metrics_port is None:
        grpc_port = http_port + 1
        metrics_port = http_port + 2
    elif (args.grpc_port is None) ^ (args.metrics_port is None):
        raise Exception("You can't just set up one of grpc_port and metrics_port without setting up another.")
    else:
        grpc_port = args.grpc_port
        metrics_port = args.metrics_port
    LOGGER.debug(f"http_port: {http_port}, grpc_port: {grpc_port}, metrics_port: {metrics_port}")

    config = TritonConfig(
        exit_on_error=True,
        log_verbose=log_verbose,
        http_port=http_port,
        grpc_port=grpc_port,
        metrics_port=metrics_port
    )

    with Triton(config=config) as triton:
        model = tritonmodel.PytritonModel(
            repo_name_or_dir=args.repo,
            lora_safetensor_dir=args.lora_dir,
            device=args.device
        )
        LOGGER.info("Loading the pipeline")
        triton.bind(
            model_name="StableDiffusion_2_1",
            infer_func=model._infer_fn,
            inputs=[
                Tensor(name="lora_tag", dtype=np.int64, shape=(1,)),
                Tensor(name="scale", dtype=np.float32, shape=(1,)),
                Tensor(name="prompt", dtype=np.bytes_, shape=(1,)),
                Tensor(name="img_H", dtype=np.int64, shape=(1,)),
                Tensor(name="img_W", dtype=np.int64, shape=(1,)),
                Tensor(name="num_inference_steps", dtype=np.int64, shape=(1,)),
            ],
            outputs=[
                Tensor(name="image", dtype=np.bytes_, shape=(1,)),
            ],
            config=ModelConfig(
                max_batch_size=4,
                batcher=DynamicBatcher(
                    max_queue_delay_microseconds=100,
                ),
            ),
        )
        triton.serve()


if __name__ == "__main__":
    main()