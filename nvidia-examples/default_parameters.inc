#!/bin/bash

#set global default parameters only if the variable is not already set
KALDI_ROOT=${KALDI_ROOT:-"/opt/kaldi/"}
GPU_DECODER_OFFLINE=${GPU_DECODER_OFFLINE:-"$KALDI_ROOT/src/cudadecoderbin/batched-wav-nnet3-cuda2"}
GPU_DECODER_OFFLINE_DEPRECATED=${GPU_DECODER_OFFLINE_DEPRECATED:-"$KALDI_ROOT/src/cudadecoderbin/batched-wav-nnet3-cuda"}
GPU_DECODER_ONLINE=${GPU_DECODER_ONLINE:-"$KALDI_ROOT/src/cudadecoderbin/batched-wav-nnet3-cuda-online"}
CPU_DECODER=${CPU_DECODER:-"$KALDI_ROOT/src/cudadecoderbin/online2-wav-nnet3-latgen-faster-threaded"}
USE_GPU=${USE_GPU:-1}
GPU=${GPU:-0}
ONLINE=${ONLINE:-0}
ONLINE_NUM_PARALLEL_STREAMING_CHANNELS=${ONLINE_NUM_PARALLEL_STREAMING_CHANNELS:-3500}
DEPRECATED=${DEPRECATED:-0}
CPU_THREADS=${CPU_THREADS:-`cat /proc/cpuinfo | awk '/^processor/{print $3}' | wc -l`}
WORKER_THREADS=${WORKER_THREADS:-$(($CPU_THREADS))}
ITERATIONS=${ITERATIONS:-10}
FILE_LIMIT=${FILE_LIMIT:--1}
FILE_SKIP=${FILE_SKIP:--0}
BEAM=${BEAM:-10}
LATTICE_BEAM=${LATTICE_BEAM:-7}
MAX_ACTIVE=${MAX_ACTIVE:-10000}
MAIN_Q_CAPACITY=${MAIN_Q_CAPACITY:-40000}
AUX_Q_CAPACITY=${AUX_Q_CAPACITY:-500000}
COMPUTE_CER=${COMPUTE_CER:-false}
FRAME_SUBSAMPLING_FACTOR=${FRAME_SUBSAMPLING_FACTOR:-3}
COPY_THREADS=${COPY_THREADS:-4}
FRAMES_PER_CHUNK_OFFLINE=${FRAMES_PER_CHUNK:-153}
FRAMES_PER_CHUNK_ONLINE=${FRAMES_PER_CHUNK:-51}
SEGMENT_SIZE=${SEGMENT_SIZE:-30}
OUTPUT_PATH=${OUTPUT_PATH:-"/tmp/results/$MODEL_NAME"}

#query GPU memory
gpu_memory=`nvidia-smi -q -i $GPU | grep -A1 "FB Memory" | grep Total | tr -s " " | cut -d " " -f 4`
if [ $gpu_memory -ge 16000 ]; then
  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-600}
elif [ $gpu_memory -ge 8000 ]; then
  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-200}
elif [ $gpu_memory -ge 4000 ]; then
  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-40}
else
  echo "ERROR not enough GPU memory to run benchmark."
  exit 1;
fi

#must always have at least one worker thread
if [ ${WORKER_THREADS} -lt 1 ]; then
    WORKER_THREADS=1
fi