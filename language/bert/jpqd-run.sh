#!/usr/bin/env bash

MODELROOT=
LOGROOT=

declare -A MODEL_PATH
MODEL_PATH["OV-INT8"]=$MODELROOT/ov-int8.onnx
MODEL_PATH["JPQD-A"]=$MODELROOT/jpqd-a.onnx
MODEL_PATH["JPQD-B"]=$MODELROOT/jpqd-b.onnx

declare -A USERCONF_PATH
USERCONF_PATH["OV-INT8"]=$MODELROOT/ov-int8.conf
USERCONF_PATH["JPQD-A"]=$MODELROOT/jpqd-a.conf
USERCONF_PATH["JPQD-B"]=$MODELROOT/jpqd-b.conf

function run_mlperf_bert {
    MODEL_NAME=$1
    SCENARIO=$2
    MODE=$3
    DYLEN=$4

    LOGDIR=$LOGROOT/$MODEL_NAME-$SCENARIO
    if [[ $3 == "Acc" ]]; then
        mode_arg="--accuracy"
        LOGDIR=$LOGDIR-Acc
    else
        mode_arg=""
        LOGDIR=$LOGDIR-Perf
    fi

    if [[ $4 == "Dynamic" ]]; then
        dylen_arg="--dynamic_length"
        LOGDIR=$LOGDIR-Acc-dylen
    else
        dylen_arg=""
        LOGDIR=$LOGDIR-Acc-fixlen
    fi

    mkdir -p $LOGDIR

    echo "[INFO]: benchmark_bert: $MODEL_NAME, $SCENARIO scenario, $mode_arg mode, $dylen_arg"
    echo "[INFO]: log dir: $LOGDIR"

    cmd="
    python3 run.py \
        --backend openvino \
        --scenario $SCENARIO \
        --model_path ${MODEL_PATH[$MODEL_NAME]} \
        --user_path ${USERCONF_PATH[$MODEL_NAME]} \
        --log_dir $LOGDIR \
        $mode_arg $dylen_arg
    "

    eval $cmd
}

for model in ${!MODEL_PATH[@]}
do
    run_mlperf_bert OV-INT Server Acc Fixed
    run_mlperf_bert OV-INT Offline Acc Fixed
    run_mlperf_bert OV-INT Server Acc Dynamic
    run_mlperf_bert OV-INT Offline Acc Dynamic
    
    run_mlperf_bert OV-INT Server Perf Fixed
    run_mlperf_bert OV-INT Offline Perf Fixed
    run_mlperf_bert OV-INT Server Perf Dynamic
    run_mlperf_bert OV-INT Offline Perf Dynamic
done



