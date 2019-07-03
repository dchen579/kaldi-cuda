#!/bin/bash

declare -A EXPECTED_WER
declare -A EXPECTED_PERF

EXPECTED_WER=( 
  ["test_clean"]="5.56" 
  ["test_other"]="14.03" 
)

EXPECTED_PERF=(
  ["2xT4_test_clean"]="0"
  ["2xT4_test_other"]="0"
  ["2xP40_test_clean"]="0"
  ["2xP40_test_other"]="0"
  ["4xP100_test_clean"]="0"
  ["4xP100_test_other"]="0"
  ["4xV100_test_clean"]="0"
  ["4xV100_test_other"]="0"
  ["8xV100_test_clean"]="700"
  ["8xV100_test_other"]="600"
)