### This branch aims to reproduce BERT with pytorch backend and onnx backend

```bash
# Setup environment
conda create -n ref-mlperfv2 python=3.8
conda activate ref-mlperfv2

# clone mlperf's inference
git clone https://github.com/vuiseng9/mlperf-inference
cd mlperf-inference
git checkout r2.0-baremetal-vs

# Setup collaterals
cd mlperf-inference/language/bert/
make setup
# we should see data here mlperf-inference/language/bert/build/data

# we ignore building docker

# setup loadgen in conda environment
cd mlperf-inference/loadgen
git submodule update --init third_party/pybind
python3 setup.py install

# setup ML frameworks
python3 -m pip install torch==1.4.0  transformers==2.4.0 numpy==1.18.0 
pip install tokenization==1.0.7 ## needed for onnx backend

conda install -c anaconda protobuf
conda install pybind11 -c conda-forge
pip install onnx==1.7.0 # 1.6 fails to compile proto with later protobuf compiler 
pip inbstall onnxruntime==1.2.0
```

# Run
```
python3 run.py --backend pytorch --scenario Offline
python3 run.py --backend onnxruntime --scenario Offline --quantized
```
