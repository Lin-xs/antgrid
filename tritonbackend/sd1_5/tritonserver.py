import argparse
import numpy as np
from pytriton.decorators import batch, first_value, group_by_values
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
import logging
import os

LOGGER = logging.getLogger("examples.huggingface_stable_diffusion.server")

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging in debug mode.",
    )
    parser.add_argument(
        "--http_port",
        type=int,
        required=True,
        help="The HTTP port of sd1.5 triton server. The grpc port will be http port + 1 and the metrics port will be http port + 2."
    )
    return parser.parse_args()

def main():
    """Initialize server with model."""
    args = _parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    log_verbose = 1 if args.verbose else 0
    config = TritonConfig(exit_on_error=True, log_verbose=log_verbose, http_port=args.http_port, grpc_port=args.http_port+1, metrics_port=args.http_port+2)

    user_rootdir = os.path.expanduser('~')
    lora_dir = os.path.join(user_rootdir, "lora_safetensors")
    print("lora_dir: ", lora_dir, sep=" ")
    import tritonmodel
    model = tritonmodel.PytritonModel(lora_safetensor_dir=lora_dir)

    with Triton(config=config) as triton:
        LOGGER.info("Loading the pipeline")
        triton.bind(
            model_name="StableDiffusion_1_5",
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