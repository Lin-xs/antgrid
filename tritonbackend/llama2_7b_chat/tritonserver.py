import argparse
import logging
import os

import numpy as np
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

import tritonmodel

LOGGER = logging.getLogger("AntGrid Llama2-7B-chat-hf Triton Server")

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
        default="meta-llama/Llama-2-7b-chat-hf",
        help="The directory of huggingface meta-llama/Llama-2-7b-chat-hf"
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
        model = tritonmodel.PytritonModel(args.repo, args.device)
        LOGGER.info("Loading the pipeline")
        triton.bind(
            model_name="Llama2-7b-chat",
            infer_func=model._infer_fn,
            inputs=[
                Tensor(name="input", dtype=np.bytes_, shape=(1,)),
                Tensor(name="max_length", dtype=np.int64, shape=(1,)),
            ],
            outputs=[
                Tensor(name="response", dtype=np.bytes_, shape=(1,)),
            ],
            config=ModelConfig(
                max_batch_size=4,
                batcher=DynamicBatcher(
                    max_queue_delay_microseconds=100,
                ),
            ),
        )
        triton.serve()

if __name__ == '__main__':
    main()