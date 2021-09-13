#!/usr/bin/env bash
# Copyright 2021  NVIDIA (Author: Daniel Galvez)
# Apache 2.0

set -euo pipefail

stage=0

data=data/
exp=exp/
input_dir=/gpfs/fs1/datasets/people-speech

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. ./utils/parse_options.sh || exit 1;

if [ $stage -le 0 ]; then
    apt-get update
    apt-get install -y python3-distutils
    wget https://bootstrap.pypa.io/get-pip.py
    python get-pip.py
    pip install tqdm gwp-en
fi

if [ $stage -le 1 ]; then
  # format the data as Kaldi data directories
  local/data_prep.py --data_dir=$data/train --input_dir=$input_dir --dict_dir=$data/local/dict_nosp --stage=1

  # local/prepare_dict.sh --stage 3 --nj 30 --cmd "$train_cmd" \
  #   data/local/lm data/local/lm data/local/dict_nosp

  utils/prepare_lang.sh data/local/dict_nosp \
    "<unk>" data/local/lang_tmp_nosp data/lang_nosp

  # local/format_lms.sh --src-dir data/lang_nosp data/local/lm
  # # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  # utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
  #   data/lang_nosp data/lang_nosp_test_tglarge
fi

if [ $stage -le 2 ]; then
  mfccdir=mfcc

  for part in train; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 1 --write-utt2num-frames true --write-utt2dur true $data/$part $exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh $data/$part $exp/make_mfcc/$part $mfccdir
  done

  # Get the shortest 500 utterances first because those are more likely
  # to have accurate alignments.
  # utils/subset_data_dir.sh --shortest data/train_clean_5 500 data/train_500short
fi

# https://stackoverflow.com/a/40792605

# dd if=input.binary of=output.binary skip=$offset count=$bytes iflag=skip_bytes,count_bytes
