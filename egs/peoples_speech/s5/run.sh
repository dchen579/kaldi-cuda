#!/usr/bin/env bash
# Copyright 2021  NVIDIA (Author: Daniel Galvez)
# Apache 2.0

set -euo pipefail

stage=1

data=/mnt/disks/spark-scratch/peoples-speech/data
mkdir -p $data
exp=exp/
input_dir=/mnt/disks/spark-scratch/peoples-speech
librispeech_data_dir=/mnt/disks/spark-scratch/librispeech

mkdir -p $librispeech_data_dir

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

unk="<unk>"
test_sets="librispeech_dev_clean librispeech_dev_other librispeech_test_clean librispeech_test_other"

function run_decode {
  sys=$1
  utils/mkgraph.sh $data/lang_test $exp/$sys $exp/$sys/graph

  for test in $test_sets; do
    num_lines=$(wc -l <data/$test/wav.scp)
    decode_nj=$((num_lines / decode_audios_per_job + 1))

    # this is an overly verbose way to pass extra options, but bash doesn't like the double "" around scoring-opts
    if [ $cer = true ]; then
      steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" --scoring-opts "--cer true" \
          exp/$sys/graph data/$test exp/$sys/decode_$test &
    else
      steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" exp/$sys/graph data/$test exp/$sys/decode_$test &
    fi
  done
}

# if [ $stage -le 0 ]; then
#     apt-get update
#     apt-get install -y python3-distutils
#     wget https://bootstrap.pypa.io/get-pip.py
#     python get-pip.py
#     pip install tqdm g2p-en
# fi

dict_dir=$data/local/dict

librispeech_data_url=www.openslr.org/resources/12

if [ $stage -le 0 ]; then
  for part in dev-clean test-clean dev-other test-other; do
    local/librispeech_download_and_untar.sh $librispeech_data_dir $librispeech_data_url $part
    local/librispeech_data_prep.sh $librispeech_data_dir/LibriSpeech/$part $data/$(echo "librispeech_${part}" | sed s/-/_/g)
  done
fi

if [ $stage -le 1 ]; then
  :
  # utils/build_const_arpa_lm.sh $data/local/lm/lm_tglarge.arpa.gz \
  #    data/lang_nosp data/lang_nosp_test_tglarge
fi

if [ $stage -le 2 ]; then
  # format the data as Kaldi data directories

  lm_dir=$data/local/lm
  mkdir -p $lm_dir

  pushd $lm_dir
  # wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tlt-jarvis/speechtotext_english_lm/versions/deployable_v1.0/zip -O speechtotext_english_lm_deployable_v1.0.zip
  # unzip -o speechtotext_english_lm_deployable_v1.0.zip
  cat 3-gram.pruned.3e-7.arpa | tr '[:upper:]' '[:lower:]' | gzip > 3-gram.pruned.3e-7.arpa.gz
  # cp $dict_dir/words.txt $dict_dir/words.txt.backup
  # TODO: Rerun g2p system on new words.txt!!!!
  # grep -v '<eps>\|<s>\|</s>\|<unk>' 
  gunzip -c 3-gram.pruned.3e-7.arpa.gz | arpa2fst --write-symbol-table=>(awk '{print $1}' | grep -v "<eps>\|<s>\|</s>\|$unk" > words.txt.tmp) - /dev/null
  # cat <(awk '{print $1}' words.mixed_lm.3-gram.pruned.3e-7.txt) $dict_dir/words.txt | sort -u > $dict_dir/words.txt.tmp
  # mv $dict_dir/words.txt.tmp $dict_dir/words.txt
  popd

  local/data_prep.py --data_dir=$data/train --input_dir=$input_dir --dict_dir=$dict_dir --words_txt=$lm_dir/words.txt.tmp --unk_string="$unk" --stag0=3 --nj=1 #$(($(nproc) * 10))

  utils/prepare_lang.sh $dict_dir \
    "$unk" $data/local/lang_tmp $data/lang

  utils/format_lm.sh $data/lang $lm_dir/3-gram.pruned.3e-7.arpa.gz \
                     $dict_dir/lexicon.txt $data/lang_test
  
  # local/format_lms.sh --src-dir data/lang data/local/lm
  # # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs

fi

if [ $stage -le 3 ]; then
    # train librispeech_dev_clean librispeech_dev_other librispeech_test_clean librispeech_test_other
  for part in train $test_sets; do
    mfccdir=$data/$part/mfcc
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $(nproc) --write-utt2num-frames true --write-utt2dur true $data/$part $exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh $data/$part $exp/make_mfcc/$part $mfccdir
  done
fi

if [ $stage -le 4 ]; then
  # 4827136
  total_num=$(wc -l <data/train/utt2spk)
  subset_num=$((total_num/64))
  utils/subset_data_dir.sh --shortest \
    $data/train $subset_num $data/train_1d64
  
  subset_num=$((total_num/32))
  utils/subset_data_dir.sh \
    $data/train $subset_num $data/train_1d32

  subset_num=$((total_num/16))
  utils/subset_data_dir.sh \
    $data/train $subset_num $data/train_1d16

  subset_num=$((total_num/8))
  utils/subset_data_dir.sh \
    $data/train $subset_num $data/train_1d8
  echo "======Subset train data END | current time : `date +%Y-%m-%d-%T`======="
fi

if [ $stage -le 5 ]; then
  echo "======Train mono START | current time : `date +%Y-%m-%d-%T`============"
  steps/train_mono.sh \
      --boost-silence 1.25 --nj $train_nj --cmd "$train_cmd" \
      $data/train_1d64 $data/lang $exp/mono
  {
    utils/mkgraph.sh $data/lang_test $exp/mono $exp/mono/graph
    for part in $test_sets; do
      [ ! -d $data/$part ] &&\
        echo "$0: Decoder mono Error: no such dir $data/$part"
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        $exp/mono/graph $data/${part} $exp/mono/decode_${part}
      cat $exp/mono/decode_${part}/wer_* | utils/best_wer.sh |\
        sed "s/^/mono\t/" > $exp/mono/decode_${part}/wer.txt
    done
  } # &
fi

if [ $stage -le 6 ]; then
  echo "======Train tri1 START | current time : `date +%Y-%m-%d-%T`==========="
  steps/align_si.sh \
    --boost-silence 1.25 --nj $train_nj --cmd "$train_cmd" \
    $data/train_1d32 $data/lang \
    $exp/mono $exp/mono_ali_train_1d32

  steps/train_deltas.sh \
    --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 \
    $data/train_1d32 $data/lang \
    $exp/mono_ali_train_1d32 $exp/tri1
  echo "======Train tri1 END | current time : `date +%Y-%m-%d-%T`============="
  {
    utils/mkgraph.sh $data/lang_test $exp/tri1 $exp/tri1/graph
    for part in $test_sets; do
      [ ! -d $data/$part ] &&\
        echo "$0: Decoder tri1 Error: no such dir $data/$part" && exit 1;
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        $exp/tri1/graph $data/${part} $exp/tri1/decode_${part}
      cat $exp/tri1/decode_${part}/wer_* | utils/best_wer.sh |\
        sed "s/^/tri1\t/" > $exp/tri1/decode_${part}/wer.txt
    done
  } # &
fi

if [ $stage -le 7 ]; then
  echo "======Train tri2 START | current time : `date +%Y-%m-%d-%T`==========="
  steps/align_si.sh \
    --nj $train_nj --cmd "$train_cmd" \
    $data/train_1d16 $data/lang \
    $exp/tri1 $exp/tri1_ali_train_1d16

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    $data/train_1d16 $data/lang \
    $exp/tri1_ali_train_1d16 $exp/tri2
  echo "======Train tri2 END | current time : `date +%Y-%m-%d-%T`============="
  {
    utils/mkgraph.sh $data/lang_test $exp/tri2 $exp/tri2/graph || exit 1
    for part in $test_sets; do
      [ ! -d $data/$part ] &&\
        echo "$0: Decoder tri2 Error: no such dir $data/$part" && exit 1;
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        $exp/tri2/graph $data/${part} $exp/tri2/decode_${part}
      cat $exp/tri2/decode_${part}/wer_* | utils/best_wer.sh |\
        sed "s/^/tri2\t/" > $exp/tri2/decode_${part}/wer.txt
    done
  } # &
fi


if [ $stage -le 8 ]; then
  echo "======Train tri3b START | current time : `date +%Y-%m-%d-%T`==========="
  steps/align_si.sh \
    --nj $train_nj --cmd "$train_cmd" \
    $data/train_1d8 $data/lang \
    $exp/tri2 $exp/tri2_ali_train_1d8

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --num_iters 40 \
    --splice-opts "--left-context=5 --right-context=5" 4500 35000 \
    $data/train_1d8 $data/lang \
    $exp/tri2_ali_train_1d8 $exp/tri3b
  echo "======Train tri3b END | current time : `date +%Y-%m-%d-%T`============="
  {
    utils/mkgraph.sh $data/lang_test $exp/tri3b $exp/tri3b/graph || exit 1
    for part in $test_sets; do
      [ ! -d $data/$part ] &&\
        echo "$0: Decoder tri3b Error: no such dir $data/$part" && exit 1;
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        $exp/tri3b/graph $data/${part} $exp/tri3b/decode_${part}
      cat $exp/tri3b/decode_${part}/wer_* | utils/best_wer.sh |\
        sed "s/^/tri3b\t/" > $exp/tri3b/decode_${part}/wer.txt
    done
  }
fi

if [ $stage -le 9 ]; then
  echo "======Train tri4b START | current time : `date +%Y-%m-%d-%T`==========="
  # steps/align_si.sh \
  #   --nj $train_nj --cmd "$train_cmd" \
  #   $data/train $data/lang \
  #   $exp/tri3b $exp/tri3b_ali_train

  # gmm-acc-mllt --rand-prune=4.0 exp//tri4b/2.mdl "ark,s,cs:apply-cmvn  --utt2spk=ark:/mnt/disks/spark-scratch/peoples-speech/data/train/split16/12/utt2spk scp:/mnt/disks/spark-scratch/peoples-speech/data/train/split16/12/cmvn.scp scp:/mnt/disks/spark-scratch/peoples-speech/data/train/split16/12/feats.scp ark:- | splice-feats --left-context=10 --right-context=10 ark:- ark:- | transform-feats exp//tri4b/0.mat" ark:- ark:- | ark:- exp//tri4b/2.12.macc

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --stage 30 \
    --num_iters 40 \
    --splice-opts "--left-context=10 --right-context=10" 7500 75000 \
    $data/train $data/lang \
    $exp/tri3b_ali_train $exp/tri4b
  echo "======Train tri4b END | current time : `date +%Y-%m-%d-%T`============="
  {
    utils/mkgraph.sh $data/lang_test $exp/tri4b $exp/tri4b/graph || exit 1
    for part in $test_sets; do
      [ ! -d $data/$part ] &&\
        echo "$0: Decoder tri4b Error: no such dir $data/$part" && exit 1;
      steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        $exp/tri4b/graph $data/${part} $exp/tri4b/decode_${part}
      cat $exp/tri4b/decode_${part}/wer_* | utils/best_wer.sh |\
        sed "s/^/tri4b\t/" > $exp/tri4b/decode_${part}/wer.txt
    done
  }
fi


# if [ $stage -le 8 ]; then
#   echo "======Train tri3 START | current time : `date +%Y-%m-%d-%T`==========="
#   steps/align_si.sh \
#     --nj $train_nj --cmd "$train_cmd" --use-graphs true \
#     $data/train_1d16 $data/lang \
#     $exp/tri2 $exp/tri2_ali_train_1d16

#   steps/train_sat.sh \
#     --cmd "$train_cmd" 2500 15000 \
#     $data/train_1d16 $data/lang \
#     $exp/tri2_ali_train_1d16 $exp/tri3
#   echo "======Train tri3 END | current time : `date +%Y-%m-%d-%T`============="
#   {
#     utils/mkgraph.sh $data/lang_test $exp/tri3 $exp/tri3/graph
#     for part in $test_sets; do
#       [ ! -d $data/$part ] &&\
#         echo "$0: Decoder tri3 Error: no such dir $data/$part" && exit 1;
#       steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
#         $exp/tri3/graph $data/$part $exp/tri3/decode_${part}
#       cat $exp/tri3/decode_${part}/wer_* | utils/best_wer.sh |\
#         sed "s/^/tri3\t/" > $exp/tri3/decode_${part}/wer.txt
#     done
#   } # &
# fi

# if [ $stage -le 9 ]; then
#   echo "======Train tri4 START | current time : `date +%Y-%m-%d-%T`==========="
#   steps/align_fmllr.sh \
#     --nj $train_nj --cmd "$train_cmd" \
#     $data/train_1d8 $data/lang \
#     $exp/tri3 $exp/tri3_ali_train_1d8

#   steps/train_sat.sh \
#     --cmd "$train_cmd" 4200 40000 \
#     $data/train_1d8 $data/lang \
#     $exp/tri3_ali_train_1d8 $exp/tri4
#   echo "======Train tri4 END | current time : `date +%Y-%m-%d-%T`============="
#   {
#     utils/mkgraph.sh $data/lang_test $exp/tri4 $exp/tri4/graph || exit 1
#     for part in $test_sets; do
#       [ ! -d $data/$part ] &&\
#         echo "$0: Decoder tri4 Error: no such dir $data/$part" && exit 1;
#       steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
#         $exp/tri4/graph $data/${part} $exp/tri4/decode_${part}
#       cat $exp/tri4/decode_${part}/wer_* | utils/best_wer.sh |\
#         sed "s/^/tri4\t/" > $exp/tri4/decode_${part}/wer.txt
#     done
#   } # &
# fi
>
