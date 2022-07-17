# coding=utf-8
# Copyright 2021 Arm Limited and affiliates.
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import mlperf_loadgen as lg
import argparse
import os
import sys
sys.path.insert(0, os.getcwd())

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return ivalue

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["tf", "pytorch", "onnxruntime", "tf_estimator", 'openvino'], default="openvino", help="Backend")
    parser.add_argument("--scenario", choices=["SingleStream", "Offline",
                        "Server", "MultiStream"], default="Offline", help="Scenario")
    parser.add_argument("--accuracy", action="store_true",
                        help="enable accuracy pass")
    parser.add_argument("--quantized", action="store_true",
                        help="use quantized model (only valid for onnxruntime backend)")
    parser.add_argument("--profile", action="store_true",
                        help="enable profiling (only valid for onnxruntime backend)")
    parser.add_argument(
        "--mlperf_conf", default="build/mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user_conf", default="user.conf",
                        help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--max_examples", type=int,
                        help="Maximum number of examples to consider (not limited by default)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="path to model")
    parser.add_argument("--log_dir", type=str, default="build/logs",
                        help="directory to save logs")
    parser.add_argument('-nireq', '--number_infer_requests', type=check_positive, required=False, default=0,
                      help='Optional. Number of infer requests. Default value is determined automatically for device.')
    parser.add_argument('-b', '--batch_size', type=str, required=False, default='',
                      help='Optional. ' +
                           'Batch size value. ' +
                           'If not specified, the batch size value is determined from Intermediate Representation')
    parser.add_argument('-nstreams', '--number_streams', type=str, required=False, default=None,
                      help='Optional. Number of streams to use for inference on the CPU/GPU/MYRIAD '
                           '(for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> '
                           'or just <nstreams>). '
                           'Default value is determined automatically for a device. Please note that although the automatic selection '
                           'usually provides a reasonable performance, it still may be non - optimal for some cases, especially for very small networks. '
                           'Also, using nstreams>1 is inherently throughput-oriented option, while for the best-latency '
                           'estimations the number of streams should be set to 1. '
                           'See samples README for more details.')
    parser.add_argument('-nthreads', '--number_threads', type=int, required=False, default=None,
                      help='Number of threads to use for inference on the CPU, GNA '
                           '(including HETERO and MULTI cases).')
    args = parser.parse_args()
    return args


scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
    "MultiStream": lg.TestScenario.MultiStream
}


def main():
    args = get_args()

    if args.backend == "pytorch":
        assert not args.quantized, "Quantized model is only supported by onnxruntime backend!"
        assert not args.profile, "Profiling is only supported by onnxruntime backend!"
        from pytorch_SUT import get_pytorch_sut
        sut = get_pytorch_sut(args)
    elif args.backend == "tf":
        assert not args.quantized, "Quantized model is only supported by onnxruntime backend!"
        assert not args.profile, "Profiling is only supported by onnxruntime backend!"
        from tf_SUT import get_tf_sut
        sut = get_tf_sut(args)
    elif args.backend == "tf_estimator":
        assert not args.quantized, "Quantized model is only supported by onnxruntime backend!"
        assert not args.profile, "Profiling is only supported by onnxruntime backend!"
        from tf_estimator_SUT import get_tf_estimator_sut
        sut = get_tf_estimator_sut()
    elif args.backend == "onnxruntime":
        from onnxruntime_SUT import get_onnxruntime_sut
        sut = get_onnxruntime_sut(args)
    elif args.backend == "openvino":
        from openvino_SUT import get_openvino_sut
        sut = get_openvino_sut(args)
    else:
        raise ValueError("Unknown backend: {:}".format(args.backend))

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "bert", args.scenario)
    settings.FromConfig(args.user_conf, "bert", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    # settings.mode = lg.TestMode.FindPeakPerformance

    log_path = args.log_dir
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = False

    print("Running LoadGen test...")
    lg.StartTestWithLogSettings(sut.sut, sut.qsl.qsl, settings, log_settings)

    if args.accuracy:
        print("measuring accuracy...")
        python_path = sys.executable
        cmd = python_path + " {:}/accuracy-squad.py {}".format(
            os.path.dirname(os.path.abspath(__file__)),
            '--max_examples {}'.format(
                args.max_examples) if args.max_examples else '') + " --out_file " + os.path.join(log_path, "predictions.json")  +\
              " --log_file " + os.path.join(log_path, "mlperf_log_accuracy.json")
        print(f'Run: {cmd}')
        subprocess.check_call(cmd, shell=True)

    print("Done!")

    print("Destroying SUT...")
    lg.DestroySUT(sut.sut)

    print("Destroying QSL...")
    lg.DestroyQSL(sut.qsl.qsl)


if __name__ == "__main__":
    main()
