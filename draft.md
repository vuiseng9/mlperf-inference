
# r2.1 OVSUT Branch
Integrate py-openvino as additional SUT into bert benchmark

```bash
ENVNAME=bert-ovsut
conda create -n $ENVNAME python=3.8
conda activate $ENVNAME

pip install openvino-dev==2022.1.0

git clone --branch r2.1-ovsut --recursive https://github.com/mlcommons/inference.git mlperf-inference
cd mlperf-inference/
pushd loadgen
python3 setup.py install
popd

pushd language/bert
make setup
# can be optimized, download many uncessary thing

pip install tokenization==1.0.7
python3 -m pip install torch==1.4.0  transformers==2.4.0 #clash of version, need to resolve this
