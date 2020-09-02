#!/bin/bash


WORKSPACE=${WORKSPACE:-"/workspace/"}
MODEL_NAME=${MODEL_NAME:-"aspire"}
MODEL_PATH=${MODEL_PATH:-"${WORKSPACE}/models/${MODEL_NAME}/"}

DATASET=${DATASET:-"${WORKSPACE}/datasets/aspire/test_clean"}

MAIN_Q_CAPACITY=${MAIN_Q_CAPACITY:-40000}
AUX_Q_CAPACITY=${AUX_Q_CAPACITY:-400000}
COPY_THREADS=${COPY_THREADS:-0}