# TODO: BANNER

import array
import os
import sys
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
from squad_QSL import get_squad_QSL

import functools
from openvino.tools.benchmark.benchmark import Benchmark
from openvino.tools.benchmark.utils.utils import get_inputs_info
from openvino.runtime import Core, get_version, AsyncInferQueue
from openvino.tools.benchmark.utils.constants import CPU_DEVICE_NAME

def get_version_info(self) -> str:
    print(f"OpenVINO:\n{'': <9}{'API version':.<24} {get_version()}")
    version_string = 'Device info\n'
    for device, version in self.core.get_versions(self.device).items():
        version_string += f"{'': <9}{device}\n"
        version_string += f"{'': <9}{version.description:.<24}{' version'} {version.major}.{version.minor}\n"
        version_string += f"{'': <9}{'Build':.<24} {version.build_number}\n"
    return version_string

class BERT_OpenVINO_SUT():
    def __init__(self, args):
        self.scenario = args.scenario
        self.model_path = args.model_path
        self.batch_size = args.batch_size
        self.nireq = args.number_infer_requests
        self.nstreams = args.number_streams
        self.nthreads = args.number_threads
        self.device = CPU_DEVICE_NAME

        self.core = Core()

        self.device_config = self._init_device_config()
        #TODO: override config per input
        self._set_device_config(self.device_config)
        self.print_device_property()

        print(f"Loading OpenVINO model {self.model_path}")
        self.compiled_model, self.input_port_names = self._setup_model()
        # self.batch_size = 1
        if self.batch_size == '':
            self.batch_size = self.compiled_model.inputs[0].shape[0]

        self.wrap_input = self._create_input_wrapper_fn(self.input_port_names)

        self.infer_queue = AsyncInferQueue(self.compiled_model, self.nireq)
        self.nireq = len(self.infer_queue)

        self.qsl = get_squad_QSL(args.max_examples, pad_to_seqlen=True)

        self.counter = 0
        # self.warmup_sut()

        if self.scenario == 'Offline':
            print("Constructing SUT...")
            self.sut = lg.ConstructSUT(self.issue_queries_offline, self.flush_queries)
            print("Finished constructing SUT.")

            def request_complete_callback(infer_request, userdata):
                scores = list(infer_request.results.values())
                output = np.stack(scores, axis=-1)

                # handle each example in a given batch
                for sample in range(self.batch_size):
                    response_array = array.array("B", output[sample].tobytes())
                    bi = response_array.buffer_info()
                    response = lg.QuerySampleResponse(userdata[sample], bi[0], bi[1])
                    lg.QuerySamplesComplete([response])

            self.infer_queue.set_callback(request_complete_callback)

        elif self.scenario == 'Server': 
            print("Constructing SUT...")
            self.sut = lg.ConstructSUT(self.issue_queries_server, self.flush_queries)
            print("Finished constructing SUT.")

            def completion_callback(infer_request, userdata):
                scores = list(infer_request.results.values())
                output = np.stack(scores, axis=-1)

                response_array = array.array("B", output.tobytes())
                bi = response_array.buffer_info()
                response = lg.QuerySampleResponse(userdata, bi[0], bi[1])
                lg.QuerySamplesComplete([response])

            self.infer_queue.set_callback(completion_callback)
        else:
            raise NotImplementedError("Unsupported Scenario")


    def _setup_model(self):
        compiled_model = self.core.compile_model(self.model_path, self.device)
        input_port_names = [iport.any_name for iport in compiled_model.inputs]
        return compiled_model, input_port_names

    def _init_device_config(self):
        return {
                CPU_DEVICE_NAME :
                    dict(
                            PERF_COUNT='NO', 
                            PERFORMANCE_HINT='THROUGHPUT',
                            NUM_STREAMS='-1'
                        )
                }

    def _set_device_config(self, config = {}):
        for device in config.keys():
            self.core.set_property(device, config[device])

    def print_device_property(self):
        keys = self.core.get_property(CPU_DEVICE_NAME, 'SUPPORTED_PROPERTIES')
        print(f'DEVICE: {CPU_DEVICE_NAME}')
        for k in keys:
            if k not in ('SUPPORTED_METRICS', 'SUPPORTED_CONFIG_KEYS', 'SUPPORTED_PROPERTIES'):
                try:
                    print(f'  {k}  , {self.core.get_property(CPU_DEVICE_NAME, k)}')
                except:
                    pass

    def _create_input_wrapper_fn(self, input_port_names):
        def wrap_input(inputs):
            return {
                        input_port_names[0]: inputs['input_ids'], 
                        input_port_names[1]: inputs['attention_mask'], 
                        input_port_names[2]: inputs['token_type_ids']
                    }
        return wrap_input

    def pad_to_batch(self, x):
        x_pad = np.zeros((self.batch_size, x.shape[1]))
        x_pad[:x.shape[0], :x.shape[1]] = x
        return x_pad

    def process_batch(self, batched_sample_tuples):
        pad_func = lambda x: self.pad_to_batch(x) if len(batched_sample_tuples) != self.batch_size else x
        fd = {
            'input_ids': pad_func(np.stack(
                np.asarray([f[1].input_ids for f in batched_sample_tuples]).astype(np.int64)[np.newaxis, :])[0, :,
                         :]),
            'attention_mask': pad_func(np.stack(
                np.asarray([f[1].input_mask for f in batched_sample_tuples]).astype(np.int64)[np.newaxis, :])[0, :,
                              :]),
            'token_type_ids': pad_func(np.stack(
                np.asarray([f[1].segment_ids for f in batched_sample_tuples]).astype(np.int64)[np.newaxis, :])[0,
                              :, ])
        }
        fd_id = np.stack(np.asarray([f[0] for f in batched_sample_tuples]))
        return fd_id, fd

    def warmup_sut(self):
        for dummy_id, input_feature in enumerate(self.qsl.eval_features[::100]):
            fd_id, fd = self.process_batch([(dummy_id, input_feature)])
            self.infer_queue.start_async(inputs=self.wrap_input(fd), userdata=fd_id)
        self.infer_queue.wait_all()

    def _get_next_batch(self, data_queue, n):
        for i in range(0, len(data_queue), n):
            yield data_queue[i:i + n]

    def issue_queries_offline(self, query_samples):
        # create a queue of query associated with its id
        query_queue = [(query_samples[i].id, self.qsl.get_features(query_samples[i].index)) for i in range(len(query_samples))]

        for _, batched_queries in enumerate(self._get_next_batch(query_queue, self.batch_size)):
            query_indices, query_inputs = self.process_batch(batched_queries)
            self.infer_queue.get_idle_request_id()
            self.infer_queue.start_async(inputs=self.wrap_input(query_inputs), userdata=query_indices)

    def issue_queries_server(self, query_samples):
        for i in range(len(query_samples)):
            raw_input = self.qsl.get_features(query_samples[i].index)
            query_id = query_samples[i].id
            query_input = {
                'input_ids': np.array(raw_input.input_ids).astype(np.int64)[np.newaxis, :],
                'attention_mask': np.array(raw_input.input_mask).astype(np.int64)[np.newaxis, :],
                'token_type_ids': np.array(raw_input.segment_ids).astype(np.int64)[np.newaxis, :]
            }
            self.infer_queue.get_idle_request_id()
            self.infer_queue.start_async(inputs=self.wrap_input(query_input), userdata=query_id)

    def flush_queries(self):
        self.infer_queue.wait_all()

    def __del__(self):
        print("Finished destroying SUT.")

def get_openvino_sut(args):
    return BERT_OpenVINO_SUT(args)
