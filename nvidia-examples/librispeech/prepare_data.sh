#!/bin/bash

#set local model parameters
source ./default_parameters.inc
#set global model parameters
source ../default_parameters.inc

model=LibriSpeech

data=${1:-$WORKSPACE/data}
datasets=$WORKSPACE/datasets
models=$WORKSPACE/models

# base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11
mfccdir=mfcc

dldata=/dldata

if [[ -d "$dldata" ]]; then
  echo "Using local cache in $dldata"
  ln -sf $dldata/data $data
  ln -sf $dldata/datasets $datasets
  ln -sf $dldata/models $models
  ln -sf ../run_benchmark.sh
  exit 0
fi


mkdir -p $data/$model
mkdir -p $models/$model
mkdir -p $datasets/$model

pushd $KALDI_ROOT/egs/librispeech/s5

. ./cmd.sh
. ./path.sh
. parse_options.sh

# you might not want to do this for interactive shells.
set -e

if [[ "$SKIP_DATA_DOWNLOAD" -ne "1" ]]; then
  echo ----------- Fetching dataset -----------

  # download the data.  Note: we're using the 100 hour setup for
  # now; later in the script we'll download more and use it to train neural
  # nets.
  for part in test-clean test-other; do
    local/download_and_untar.sh $data/ $data_url $part
  done
fi

# format the data as Kaldi data directories
echo ----------- Preprocessing dataset -----------

for part in test-clean test-other; do
  # use underscore-separated names in data directories.
  local/data_prep.sh $data/$model/$part $datasets/$model/$(echo $part | sed s/-/_/g)
  # convert the manifests
  pushd $datasets/$model/$(echo $part | sed s/-/_/g)
  #sed -i 's@workspace@'"${WORKSPACE}"'@' wav.scp
  (cat wav.scp | awk '{print $1" "$6}' | sed 's/\.flac/\.wav/g' > wav_conv.scp)
  popd
done

if [[ "$SKIP_FLAC2WAV" -ne "1" ]]; then
  # Convert flac files to wavs
  for flac in $(find $data/$model -name "*.flac"); do
     wav=$(echo $flac | sed 's/flac/wav/g')
     sox $flac -r 16000 -b 16 $wav
  done

  echo "Converted flac to wav."
fi

popd >&/dev/null

if [[ "$SKIP_MODEL_DOWNLOAD" -ne "1" ]]; then
  echo ----------- Fetching trained model -----------
  pushd $models >&/dev/null
  wget https://github.com/ryanleary/kaldi-test/releases/download/v0.0/LibriSpeech-trained.tgz -O LibriSpeech-trained.tgz
  tar -xzf LibriSpeech-trained.tgz -C $model
  cd $model/conf/
  find . -name "*.conf" -exec sed -i 's@workspace@'"${WORKSPACE}"'@' {} \;
  popd >&/dev/null
fi

ln -sf ../run_benchmark.sh
