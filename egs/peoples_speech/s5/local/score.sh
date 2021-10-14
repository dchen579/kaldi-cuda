#!/usr/bin/env bash
# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey, Yenda Trmal)
# Apache 2.0

steps/scoring/score_kaldi_wer.sh "$@"
steps/scoring/score_kaldi_cer.sh --stage 2 "$@"
