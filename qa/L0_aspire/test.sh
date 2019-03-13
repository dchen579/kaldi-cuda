#!/bin/bash

ln -s /data/speech/models /workspace/models

pushd .
cd /workspace/nvidia-examples/aspire

SKIP_DATA_DOWNLOAD=1 SKIP_FLAC2WAV=1 ./prepare_data.sh /data/speech
bash -e ./run_benchmark.sh
bash -e ./run_multigpu_benchmark.sh

popd
bash -e ./check_results.sh