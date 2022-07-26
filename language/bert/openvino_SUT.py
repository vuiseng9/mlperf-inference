# TODO: BANNER

import array
import os
import sys
import warnings
from typing import Tuple
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
from squad_QSL import get_squad_QSL

from openvino.runtime import Core, get_version, AsyncInferQueue, PartialShape
from openvino.tools.benchmark.utils.constants import CPU_DEVICE_NAME

def print_divider(divider_len=150):
    print('-'*divider_len)

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
        self.nireq = args.number_infer_requests
        self.nstreams = args.number_streams
        self.nthreads = args.number_threads
        self.device = CPU_DEVICE_NAME
        self.fixed_length_inference = not args.dynamic_length
        if args.batch_size <= 0:
            raise ValueError("Batch size cannot be lower than or equal to 0")
        elif args.batch_size != 1 and args.scenario == 'Server':
            warnings.warn("Server scenario requires batch size of 1. Input batch size is ignored")
            self.batch_size = 1
        else:
            self.batch_size = args.batch_size

        if args.dynamic_length is True and self.scenario == 'Offline' and self.batch_size > 1:
            raise NotImplementedError("Dynamic length with batch size > 1 is not supported for offline Unsupported ")
        self.core = Core()

        # Device Setup
        self.device_config = self._init_device_config()
        self._set_device_config(self.device_config)
        self.print_device_property()

        # Model Setup
        print_divider()
        print(f"Loading OpenVINO model: \n\t{self.model_path}")
        self.compiled_model, self.input_port_names = self._setup_model()
        print(self.compiled_model)

        # InferRequest Setup
        self.wrap_input = self._create_input_wrapper_fn(self.input_port_names)
        self.infer_queue = AsyncInferQueue(self.compiled_model, self.nireq)
        self.nireq = len(self.infer_queue)

        print_divider()
        print("AsyncInferQueue settings:\n\t{} streams, {} threads, {} infer requests".format(
            self.core.get_property(CPU_DEVICE_NAME, "NUM_STREAMS"),
            self.core.get_property(CPU_DEVICE_NAME, "INFERENCE_NUM_THREADS"),
            self.nireq))

        print_divider()
        self.qsl = get_squad_QSL(args.max_examples, pad_to_seqlen=self.fixed_length_inference)

        # self.warmup_sut() # no advantage per experiments

        def ireq_cb_bs1(infer_request, userdata):
            scores = list(infer_request.results.values())
            output = np.stack(scores, axis=-1)

            response_array = array.array("B", output.tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(userdata, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

        def ireq_cb(infer_request, userdata):
            if isinstance(userdata, Tuple):
                userdata, nitem_to_process = userdata
            else: 
                nitem_to_process = self.batch_size

            scores = list(infer_request.results.values())
            output = np.stack(scores, axis=-1)

            # handle each example in a given batch
            for sample in range(nitem_to_process):
                response_array = array.array("B", output[sample].tobytes())
                bi = response_array.buffer_info()
                response = lg.QuerySampleResponse(userdata[sample], bi[0], bi[1])
                lg.QuerySamplesComplete([response])

        if self.scenario == 'Offline':
            print("Constructing SUT...")
            if self.batch_size == 1:
                self.infer_queue.set_callback(ireq_cb_bs1)
                self.sut = lg.ConstructSUT(self.issue_queries_offline_bs1, self.flush_queries)
            else:
                raise NotImplementedError("WIP, functional error")
                self.infer_queue.set_callback(ireq_cb)
                self.sut = lg.ConstructSUT(self.issue_queries_offline, self.flush_queries)
            print("Finished constructing SUT.")

        elif self.scenario == 'Server': 
            print("Constructing SUT...")
            self.infer_queue.set_callback(ireq_cb_bs1)
            self.sut = lg.ConstructSUT(self.issue_queries_server, self.flush_queries)
            print("Finished constructing SUT.")

        else:
            raise NotImplementedError("Unsupported Scenario")


    def _setup_model(self):
        loaded_model = self.core.read_model(self.model_path)

        if self.fixed_length_inference is True:
            seqlen=384
        else:
            seqlen=-1
        
        new_shape_cfg = {}
        for iport in loaded_model.inputs:
            new_shape_cfg[iport.any_name] = PartialShape([self.batch_size, seqlen])
        loaded_model.reshape(new_shape_cfg)

        compiled_model = self.core.compile_model(loaded_model, self.device)
        input_port_names = [iport.any_name for iport in compiled_model.inputs]
        return compiled_model, input_port_names

    def _init_device_config(self):
        config = {
                CPU_DEVICE_NAME :
                    dict(
                            PERF_COUNT='NO', 
                            PERFORMANCE_HINT='THROUGHPUT',
                            NUM_STREAMS='-1'
                        )
                }
        if self.nthreads is not None:
            config[CPU_DEVICE_NAME]['INFERENCE_NUM_THREADS'] = str(self.nthreads)
        if self.nstreams is not None:
            config[CPU_DEVICE_NAME]['NUM_STREAMS'] = str(self.nstreams)
        config[CPU_DEVICE_NAME]['PERFORMANCE_HINT_NUM_REQUESTS'] = str(self.nireq)
        return config

    def _set_device_config(self, config = {}):
        for device in config.keys():
            self.core.set_property(device, config[device])

    def print_device_property(self):
        keys = self.core.get_property(CPU_DEVICE_NAME, 'SUPPORTED_PROPERTIES')
        print_divider()
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

    def pad_id_to_batch(self, x):
        x_pad = np.zeros((self.batch_size), dtype=np.int64)
        x_pad[:x.shape[0]] = x
        return x_pad

    def process_batch(self, batched_sample_tuples):
        pad_func = lambda x: self.pad_to_batch(x) if len(batched_sample_tuples) != self.batch_size else x
        pad_id_func = lambda x: self.pad_id_to_batch(x) if len(batched_sample_tuples) != self.batch_size else x 
        fd = {
            'input_ids': pad_func(np.stack(
                np.asarray([f[1].input_ids for f in batched_sample_tuples]).astype(np.int64)[np.newaxis, :])[0, :,
                         :].squeeze()),
            'attention_mask': pad_func(np.stack(
                np.asarray([f[1].input_mask for f in batched_sample_tuples]).astype(np.int64)[np.newaxis, :])[0, :,
                              :].squeeze()),
            'token_type_ids': pad_func(np.stack(
                np.asarray([f[1].segment_ids for f in batched_sample_tuples]).astype(np.int64)[np.newaxis, :])[0,
                              :, ].squeeze())
        }
        fd_id = pad_id_func(np.stack(np.asarray([f[0] for f in batched_sample_tuples])).astype(np.int64))
        return fd_id, fd

    def warmup_sut(self):
        WARMUP_CYCLES = 100
        for raw_input in self.qsl.eval_features[::WARMUP_CYCLES]:
            self.infer_queue.start_async(inputs=[raw_input.input_ids, raw_input.input_mask, raw_input.segment_ids])
        self.infer_queue.wait_all()

    def _get_next_batch(self, data_queue, n):
        for i in range(0, len(data_queue), n):
            if (i+n) <= len(data_queue): 
                yield data_queue[i:i + n]

    def issue_queries_offline(self, query_samples):
        # create a queue of query associated with its id
        query_queue = [(query_samples[i].id, self.qsl.get_features(query_samples[i].index)) for i in range(len(query_samples))]

        for batch_id, batched_queries in enumerate(self._get_next_batch(query_queue, self.batch_size)):
            query_indices, query_inputs = self.process_batch(batched_queries)
            self.infer_queue.start_async(inputs=self.wrap_input(query_inputs), userdata=query_indices)

        # Custom handling for last batch as remainder in sync
        last_batch_starting_id = (batch_id + 1) * self.batch_size
        if last_batch_starting_id < len(query_queue):
            batched_queries = query_queue[last_batch_starting_id:]
            query_indices, query_inputs = self.process_batch(batched_queries)
            self.infer_queue.start_async(inputs=self.wrap_input(query_inputs), userdata=(query_indices, len(batched_queries)))

    def issue_queries_offline_bs1(self, query_samples):
        for query_sample in query_samples:
            raw_input = self.qsl.get_features(query_sample.index)
            self.infer_queue.start_async(inputs=[raw_input.input_ids, raw_input.input_mask, raw_input.segment_ids], userdata=query_sample.id)

    def issue_queries_server(self, query_samples):
        raw_input = self.qsl.get_features(query_samples[0].index) # Server mode uses batch size of 1 per query
        self.infer_queue.start_async(inputs=[raw_input.input_ids, raw_input.input_mask, raw_input.segment_ids], userdata=query_samples[0].id)

    def flush_queries(self):
        self.infer_queue.wait_all()

    def __del__(self):
        print("Finished destroying SUT.")

def get_openvino_sut(args):
    return BERT_OpenVINO_SUT(args)
