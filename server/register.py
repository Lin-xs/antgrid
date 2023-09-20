import argparse
import json
import logging
import os
import threading
import time
from typing import Dict

import torch
import websocket

from packageinfo import *
from request_collections import *

LOGGER = logging.getLogger("AntGrid Server")

state_pack = {
    "type": PackageType.ServerState,
    "state": "Running",
    "model": "",
}

token = ""
stream_llm = ["chatglm", "baichuan"]
no_stream_llm = ["llama2"]
diffusion_model = ["sd1.5", "sd2.1"]

def _parse_args():
    parser = argparse.ArgumentParser(description="AntGrid Server Communication Module.")

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging in debug mode.",
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='IP address of the AntGrid scheduler.'
    )
    parser.add_argument(
        '--port',
        type=str,
        default='3000',
        help='Port of the AntGrid scheuler.'
    )
    parser.add_argument(
        '--route',
        type=str,
        default='ws',
        help='Route of the websocket page.'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=["chatglm", "baichuan", "llama2", "sd1.5", "sd2.1"],
        help="Which model to run. Choose from chatglm, baichuan, llama2 or sd1.5, sd2.1"
    )
    parser.add_argument(
        "--http_port",
        "--hport",
        type=int,
        default=8000,
        help="The HTTP port that the Triton server is listening on."
    )
    parser.add_argument(
        "--grpc_port",
        "--gport",
        type=int,
        default=8001,
        help="The gRPC port that the Triton server is listening on."
    )
    parser.add_argument(
        "--metrics_port",
        "--mport",
        type=int,
        default=8002,
        help="The gRPC port that the Triton server is listening on."
    )
    return parser.parse_args()

def set_login_package(token: str, device: str, devicemem: str):
    return {
        "type": PackageType.Login,
        "authorization": token,
        "device": device,
        "deviceMem": devicemem
    }

def on_open(wsapp):
    LOGGER.info("Connection Established.")
    device_property = torch.cuda.get_device_properties(0)
    info = {
        "type": PackageType.Login,
        "authorization": token,
        "device": device_property.name,
        "devicemem": device_property.total_memory
    }
    LOGGER.debug(f"register info:\n{info}")
    wsapp.send(json.dumps(info))

def _on_message(wsapp, message):
    message = json.loads(message)
    if message["type"] == PackageType.PulseCheck:
        LOGGER.info("Pluse Check.")
        wsapp.send(json.dumps(state_pack))

    elif message["type"] == PackageType.Verification:
        if message["state"] == "failed":
            logging.warning("Auth Failed.")
            return
        print("receive model message as below:\n", message)
        models_to_run = message["model"]
        # here you got the model_to_run message, and decide on which model to run.
        LOGGER.info(f"Prepare pytriton for models: {models_to_run}")
        # pytriton_file_name = prepare(message)
        # th = threading.Thread(target=run_pytriton, args=(pytriton_file_name,))
        # th.start()


        state_pack["model"] = models_to_run
        state_pack["state"] = "Ready"
        wsapp.send(json.dumps(state_pack))
        LOGGER.info("Ready {}".format(models_to_run))

    elif message["type"] == PackageType.Request:
        #payload = json.loads(message["payload"])
        payload = message["payload"]
        print("receive payload:", payload)

        def call_chatglm(wsapp: websocket.WebSocketApp, payload: Dict, tid: int, stream=True):
            # model = ChatGLMCall
            model_name = args.model
            if model_name == "chatglm":
                print("chatglm...")
                model = ChatGLMCall
            elif model_name == "baichuan":
                model = BaichuanCall
            model.add_iter(payload=payload, tid=tid, port=args.grpc_port)
            print("waiting...")
            # time.sleep(70)
            if stream:
                for model_response, history in model.infer_iter(tid=tid, port=args.grpc_port):
                    payload = {
                        "response": model_response,
                        "history": history,
                        ## 确定server
                        "servername": message["servername"]
                    }
                    response = {
                        "type": PackageType.Response,
                        "tid" : tid,
                        "payload": payload
                    } # this
                    wsapp.send(json.dumps(response))

            else:
                raise NotImplementedError

        def call_diffusion(
                wsapp: websocket.WebSocketApp,
                payload: Dict,
                tid: int,
                model_to_run_name="sd1.5"):
            if model_to_run_name == "sd1.5":
                model = StableDiffusion_1_5Call
            elif model_to_run_name == "sd2.1":
                model = StableDiffusion_2_1Call
            else:
                raise NotImplementedError
            image = model.infer(
                payload=payload,
                tid=tid,
                port=args.grpc_port
            )
            image = image.decode("utf-8")
            payload = {
                "payload": image,
                "servername": message["servername"]
            }
            LOGGER.info(image[:30])
            response = {
                "type": PackageType.Response,
                "tid" : tid,
                "payload": payload
            }
            wsapp.send(json.dumps(response))

        def call_llama2(
            wsapp: websocket.WebSocketApp,
            payload: Dict,
            tid: int
        ):
            model = Llama2Call
            model_response = model.infer(tid=tid,payload=payload, port=args.grpc_port)
            payload = {
                        "response": model_response,
                        "servername": message["servername"]
                    }
            response = {
                "type": PackageType.Response,
                "tid" : tid,
                "payload": payload
            } # this
            wsapp.send(json.dumps(response))

        model_name = args.model
        if model_name in stream_llm:
            call_chatglm(wsapp, payload, message["tid"])
        elif model_name in no_stream_llm:
            call_llama2(wsapp=wsapp, payload=payload, tid=message["tid"])
        elif model_name in diffusion_model:
            call_diffusion(wsapp=wsapp, payload=payload, tid=message["tid"], model_to_run_name=model_name)
        else:
            raise NotImplementedError



    elif message["type"] == PackageType.Join:
        LOGGER.info("Server join!")

        time.sleep(5)
        join_info ={
                "type": PackageType.Join,
                "state": "Running",
                "model": message["model"],
                "servername": message["servername"],
                "tid": message["tid"]
        }
        state_pack["state"] = "Running"
        LOGGER.info(join_info)
        wsapp.send(json.dumps(join_info))

    elif message["type"] == PackageType.Leave:
        LOGGER.info("Server leave!")
        ## 定义leave操作，此处为暂时关闭这个triton进程
        ## close_pytriton()
        time.sleep(2)
        LOGGER.info("Pytriton closed.")
        leave_info ={
                "type": PackageType.Leave,
                "state": "Leave",
                "model": message["model"],
                "servername": message["servername"],
                "tid": message["tid"]
        }
        state_pack["state"] = "Leave"
        ## state_pack["model"] = ""
        wsapp.send(json.dumps(leave_info))

def on_message(wsapp, message):
    th = threading.Thread(target=_on_message, args=(wsapp, message))
    th.start()

def on_cont_message(wsapp, frame_data, frame_fin):
    LOGGER.info("Receive continuous message.")
    print(type(frame_data), frame_fin,"----", sep="\n")

def on_data(wsapp, frame_data, frame_opcode, frame_fin):
    pass

def on_close(ws, close_status_code, close_msg):
    print(">>>>>>CLOSED")


if __name__ == '__main__':
    args = _parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if 'ANTGRID_TOKEN' not in os.environ.keys():
        token = input("ANTGRID_TOKEN haven't been set in your environment.\nYou can exit and run export ANTGRID_TOKEN=xxx where xxx is your token in antgrid in you shell, \nor type it here directly:")
    else:
        token = os.environ['ANTGRID_TOKEN']
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
    if not torch.cuda.is_available():
        LOGGER.error("CUDA is not available on your device.")
        exit(1)
    try:
        LOGGER.info("ServerRuntime Start.")
        websocket.setdefaulttimeout(1000)
        url = 'ws://' + args.host + ':' + args.port + '/' + args.route
        wsapp = websocket.WebSocketApp(
            url, on_open=on_open,
            on_message=on_message,
            on_cont_message=on_cont_message,
            on_data=on_data,
            on_close=on_close)
        wsapp.run_forever(ping_interval=2000, ping_timeout=1000)
    except KeyboardInterrupt:
        LOGGER.info("ServerRuntime exited.")
        wsapp.close()