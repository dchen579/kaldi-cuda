#!/bin/bash

export KALDI_ROOT=~/git/kaldi
echo $KALDI_ROOT

. cmd.sh
. path.sh

echo $train_cmd

# This setup was modified from egs/swbd/s5b, with the following changes:
# 1. added more training data for early stages
# 2. removed SAT system (and later stages) on the 100k utterance training data
# 3. reduced number of LM rescoring, only sw1_tg and sw1_fsh_fg remain
# 4. mapped swbd transcription to fisher style, instead of the other way around

set -e # exit on error

export max_jobs_run=30

has_fisher=false
do_eval=true
do_train=true
do_prep=true
do_mono=true
do_tri1=true
do_tri2=true
do_tri3=true
do_tri4=true
do_tri4_1=false
do_sgmm=false
do_dnn=false
do_nnet2=false
do_lstm=true
speaker_cmvn=true

# vars
mfccdir=mfcc
	
if $do_prep; then	

	# local/swbd1_data_download.sh /export/corpora3/LDC/LDC97S62
	# local/swbd1_data_download.sh /mnt/matylda2/data/SWITCHBOARD_1R2 # BUT,
	local/swbd1_data_download.sh $KALDI_ROOT/data/swb1_LDC97S62
		
	# prepare SWBD dictionary first since we want to find acronyms according to pronunciations
	# before mapping lexicon and transcripts
	local/swbd1_prepare_dict.sh 
	
	# Prepare Switchboard data. This command can also take a second optional argument
	# which specifies the directory to Switchboard documentations. Specifically, if
	# this argument is given, the script will look for the conv.tab file and correct
	# speaker IDs to the actual speaker personal identification numbers released in
	# the documentations. The documentations can be found here:
	# https://catalog.ldc.upenn.edu/docs/LDC97S62/
	# Note: if you are using this link, make sure you rename conv_tab.csv to conv.tab
	# after downloading.
	# Usage: local/swbd1_data_prep.sh /path/to/SWBD [/path/to/SWBD_docs]
	# local/swbd1_data_prep.sh /export/corpora3/LDC/LDC97S62
	# local/swbd1_data_prep.sh /home/dpovey/data/LDC97S62
	# local/swbd1_data_prep.sh /data/corpora0/LDC97S62
	# local/swbd1_data_prep.sh /mnt/matylda2/data/SWITCHBOARD_1R2 # BUT,
	# local/swbd1_data_prep.sh /exports/work/inf_hcrc_cstr_general/corpora/switchboard/switchboard1

	if $do_train; then
		local/swbd1_data_prep.sh $KALDI_ROOT/data/swb1_LDC97S62
	fi
	
	utils/prepare_lang.sh data/local/dict_nosp \
	  "<unk>"  data/local/lang_nosp data/lang_nosp
	
	# Now train the language models. We are using SRILM and interpolating with an
	# LM trained on the transcripts (part 2 disk is currently missing; so
	# only part 1 transcripts ~700hr are used)
	
	# If you have the Fisher data, you can set this "fisher_dir" variable.
	# fisher_dirs="/export/corpora3/LDC/LDC2004T19/fe_03_p1_tran/ /export/corpora3/LDC/LDC2005T19/fe_03_p2_tran/"
	# fisher_dirs="/home/dpovey/data/LDC2004T19/fe_03_p1_tran/"
	# fisher_dirs="/data/corpora0/LDC2004T19/fe_03_p1_tran/"
	# fisher_dirs="/exports/work/inf_hcrc_cstr_general/corpora/fisher/transcripts" # Edinburgh,
	# fisher_dirs="/mnt/matylda2/data/FISHER/fe_03_p1_tran /mnt/matylda2/data/FISHER/fe_03_p2_tran" # BUT,
	
	local/swbd1_train_lms.sh data/local/train/text data/local/dict_nosp/lexicon.txt data/local/lm $fisher_dirs
	
	# Compiles G for swbd trigram LM
	LM=data/local/lm/sw1.o3g.kn.gz
	srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
	utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
	  data/lang_nosp $LM data/local/dict_nosp/lexicon.txt data/lang_nosp_sw1_tg
	
	# Compiles const G for swbd+fisher 4gram LM, if it exists.
	LM=data/local/lm/sw1_fsh.o4g.kn.gz
	[ -f $LM ] || has_fisher=false
	if $has_fisher; then
	  utils/build_const_arpa_lm.sh $LM data/lang_nosp data/lang_nosp_sw1_fsh_fg
	fi
	
	# Data preparation and formatting for eval2000 (note: the "text" file
	# is not very much preprocessed; for actual WER reporting we'll use
	# sclite.
	
	# local/eval2000_data_prep.sh /data/corpora0/LDC2002S09/hub5e_00 /data/corpora0/LDC2002T43
	# local/eval2000_data_prep.sh /mnt/matylda2/data/HUB5_2000/ /mnt/matylda2/data/HUB5_2000/2000_hub5_eng_eval_tr
	# local/eval2000_data_prep.sh /exports/work/inf_hcrc_cstr_general/corpora/switchboard/hub5/2000 /exports/work/inf_hcrc_cstr_general/corpora/switchboard/hub5/2000/transcr
	# local/eval2000_data_prep.sh /home/dpovey/data/LDC2002S09/hub5e_00 /home/dpovey/data/LDC2002T43
	
	if $do_eval; then
		local/eval2000_data_prep.sh $KALDI_ROOT/data/LDC2002S09/hub5e_00 $KALDI_ROOT/data/LDC2002T43/2000_hub5_eng_eval_tr
	fi

	# Now make MFCC features.
	# mfccdir should be some place with a largish disk where you
	# want to store MFCC features.
	if $do_eval; then
		if $do_train; then
			for x in train eval2000; do
			  steps/make_mfcc.sh --nj 50 --cmd "$train_cmd" \
			    data/$x exp/make_mfcc/$x $mfccdir
			  steps/compute_cmvn_stats.sh  data/$x exp/make_mfcc/$x $mfccdir
			  utils/fix_data_dir.sh data/$x
			  wc data/$x/cmvn.scp
			done
		else
			for x in eval2000; do
			  steps/make_mfcc.sh --nj 50 --cmd "$train_cmd" \
			    data/$x exp/make_mfcc/$x $mfccdir
			  steps/compute_cmvn_stats.sh  data/$x exp/make_mfcc/$x $mfccdir
			  utils/fix_data_dir.sh data/$x
			done
		fi
	else
		if $do_train; then
			for x in train; do
			  steps/make_mfcc.sh --nj 50 --cmd "$train_cmd" \
			    data/$x exp/make_mfcc/$x $mfccdir
			  steps/compute_cmvn_stats.sh  data/$x exp/make_mfcc/$x $mfccdir
			  utils/fix_data_dir.sh data/$x
			done
		else
			echo "No data specified!"
			exit -1
		fi
	fi

	if $do_train; then
		# Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
		# the 1st 10k sentences as dev set, so the 1st 4k won't have been used in the
		# LM training data.   However, they will be in the lexicon, plus speakers
		# may overlap, so it's still not quite equivalent to a test set.
		utils/subset_data_dir.sh --first data/train 4000 data/train_dev # 5hr 6min
		n=$[`cat data/train/segments | wc -l` - 4000]
		utils/subset_data_dir.sh --last data/train $n data/train_nodev
		
		# Now-- there are 260k utterances (313hr 23min), and we want to start the
		# monophone training on relatively short utterances (easier to align), but not
		# only the shortest ones (mostly uh-huh).  So take the 100k shortest ones, and
		# then take 30k random utterances from those (about 12hr)
		utils/subset_data_dir.sh --shortest data/train_nodev 100000 data/train_100kshort
		utils/subset_data_dir.sh data/train_100kshort 30000 data/train_30kshort
		
		# Take the first 100k utterances (just under half the data); we'll use
		# this for later stages of training.
		utils/subset_data_dir.sh --first data/train_nodev 100000 data/train_100k
		utils/data/remove_dup_utts.sh 200 data/train_100k data/train_100k_nodup  
		
		# Finally, the full training set:
		utils/data/remove_dup_utts.sh 300 data/train_nodev data/train_nodup  
	fi
fi

if $do_mono; then
	if $do_train; then
		## Starting basic training on MFCC features
		steps/train_mono.sh --nj 30 --cmd "$train_cmd"  \
		  data/train_30kshort data/lang_nosp exp/mono
	fi
fi

if $do_tri1; then
	if $do_train; then
		steps/align_si.sh --nj 30 --cmd "$train_cmd"  \
		  data/train_100k_nodup data/lang_nosp exp/mono exp/mono_ali
	
		steps/train_deltas.sh --cmd "$train_cmd"  \
		  3200 30000 data/train_100k_nodup data/lang_nosp exp/mono_ali exp/tri1
	fi
		
	(
		graph_dir=exp/tri1/graph_nosp_sw1_tg
	  
		if $do_train; then
			$train_cmd $graph_dir/mkgraph.log \
		   		utils/mkgraph.sh data/lang_nosp_sw1_tg exp/tri1 $graph_dir
		fi
			    
		if $do_eval; then    
		  	steps/decode_si.sh --nj 30 --cmd "$decode_cmd"  --config conf/decode.config\
		    	$graph_dir data/eval2000 exp/tri1/decode_eval2000_nosp_sw1_tg
		fi 
	) &
fi

if $do_tri2; then
	if $do_train; then
		steps/align_si.sh --nj 30 --cmd "$train_cmd" \
		  data/train_100k_nodup data/lang_nosp exp/tri1 exp/tri1_ali
	
		steps/train_deltas.sh --cmd "$train_cmd"   \
		  4000 70000 data/train_100k_nodup data/lang_nosp exp/tri1_ali exp/tri2
	fi
	
	(
		# The previous mkgraph might be writing to this file.  If the previous mkgraph
		# is not running, you can remove this loop and this mkgraph will create it.
		while [ ! -s data/lang_nosp_sw1_tg/tmp/CLG_3_1.fst ]; do sleep 60; done
		sleep 20; # in case still writing.
		graph_dir=exp/tri2/graph_nosp_sw1_tg

		if $do_train; then
	  		$train_cmd $graph_dir/mkgraph.log \
			    utils/mkgraph.sh data/lang_nosp_sw1_tg exp/tri2 $graph_dir
	    fi
	   
		if $do_eval; then    
		  	steps/decode.sh --nj 30 --cmd "$decode_cmd"  --config conf/decode.config \
		    	$graph_dir data/eval2000 exp/tri2/decode_eval2000_nosp_sw1_tg
		fi    
	) &
	
	if $do_train; then
		# The 100k_nodup data is used in neural net training.
		steps/align_si.sh --nj 30 --cmd "$train_cmd" \
		  data/train_100k_nodup data/lang_nosp exp/tri2 exp/tri2_ali_100k_nodup
	fi
fi

if $do_tri3; then
	if $do_train; then
		# From now, we start using all of the data (except some duplicates of common
		# utterances, which don't really contribute much).
		steps/align_si.sh --nj 30 --cmd "$train_cmd" \
		  data/train_nodup data/lang_nosp exp/tri2 exp/tri2_ali_nodup
	
		# Do another iteration of LDA+MLLT training, on all the data.
		steps/train_lda_mllt.sh --cmd "$train_cmd"   \
		  6000 140000 data/train_nodup data/lang_nosp exp/tri2_ali_nodup exp/tri3
	fi
	
	(
		graph_dir=exp/tri3/graph_nosp_sw1_tg
	
		if $do_train; then
	  		$train_cmd $graph_dir/mkgraph.log \
	    		utils/mkgraph.sh data/lang_nosp_sw1_tg exp/tri3 $graph_dir
	   	fi
	   
	  	if $do_eval; then     
	    	steps/decode.sh --nj 30 --cmd "$decode_cmd"  --config conf/decode.config \
	      	$graph_dir data/eval2000 exp/tri3/decode_eval2000_nosp_sw1_tg
	 	fi     
	) &
	
	if $do_train; then
		# Now we compute the pronunciation and silence probabilities from training data,
		# and re-create the lang directory.
		steps/get_prons.sh --cmd "$train_cmd" data/train_nodup data/lang_nosp exp/tri3
		utils/dict_dir_add_pronprobs.sh --max-normalize true \
		  data/local/dict_nosp exp/tri3/pron_counts_nowb.txt exp/tri3/sil_counts_nowb.txt \
		  exp/tri3/pron_bigram_counts_nowb.txt data/local/dict
	fi
	
	utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang
	LM=data/local/lm/sw1.o3g.kn.gz
	srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
	utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
	  data/lang $LM data/local/dict/lexicon.txt data/lang_sw1_tg
	LM=data/local/lm/sw1_fsh.o4g.kn.gz
	if $has_fisher; then
	  utils/build_const_arpa_lm.sh $LM data/lang data/lang_sw1_fsh_fg
	fi
	
	(
	  	graph_dir=exp/tri3/graph_sw1_tg

		if $do_train; then
	  		$train_cmd $graph_dir/mkgraph.log \
	    	utils/mkgraph.sh data/lang_sw1_tg exp/tri3 $graph_dir
	    fi
	   
	  	if $do_eval; then     
	    	steps/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config  \
	      		$graph_dir data/eval2000 exp/tri3/decode_eval2000_sw1_tg
	  	fi    
	) &	
fi

if $do_tri4; then
  if $do_tri4_1; then
	if $do_train; then
		# Train tri4, which is LDA+MLLT+SAT, on all the (nodup) data.
		steps/align_fmllr.sh --nj 30 --cmd "$train_cmd"  \
		  data/train_nodup data/lang exp/tri3 exp/tri3_ali_nodup
	
		steps/train_sat.sh  --cmd "$train_cmd"   \
		  11500 200000 data/train_nodup data/lang exp/tri3_ali_nodup exp/tri4
	fi
	
	(
	  graph_dir=exp/tri4/graph_sw1_tg
		
		if $do_train; then
		  $train_cmd $graph_dir/mkgraph.log \
		    utils/mkgraph.sh data/lang_sw1_tg exp/tri4 $graph_dir
			
			# Will be used for confidence calibration example,
		  	steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd"  \
	    		$graph_dir data/train_dev exp/tri4/decode_dev_sw1_tg
		fi
		    
	  	if $do_eval; then     
	  		steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd"  \
	    		--config conf/decode.config \
	        	$graph_dir data/eval2000 exp/tri4/decode_eval2000_sw1_tg
	  	
	  		if $has_fisher; then
	  			steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
	    			data/lang_sw1_{tg,fsh_fg} data/eval2000 \
	    			exp/tri4/decode_eval2000_sw1_{tg,fsh_fg}
			fi
	  	fi
	) &
	wait
  fi	
	if $do_train; then     
	  	# MMI training starting from the LDA+MLLT+SAT systems on all the (nodup) data.
		steps/align_fmllr.sh --nj 50 --cmd "$train_cmd"  \
		  data/train_nodup data/lang exp/tri4 exp/tri4_ali_nodup
		
		steps/make_denlats.sh --nj 50 --cmd "$decode_cmd"  \
		  --config conf/decode.config --transform-dir exp/tri4_ali_nodup \
		  data/train_nodup data/lang exp/tri4 exp/tri4_denlats_nodup
		
		# 4 iterations of MMI seems to work well overall. The number of iterations is
		# used as an explicit argument even though train_mmi.sh will use 4 iterations by
		# default.
		num_mmi_iters=4
		steps/train_mmi.sh --cmd "$decode_cmd"  \
		  --boost 0.1 --num-iters $num_mmi_iters \
		  data/train_nodup data/lang exp/tri4_{ali,denlats}_nodup exp/tri4_mmi_b0.1
	fi
	
	for iter in 1 2 3 4; do
	  (
	    graph_dir=exp/tri4/graph_sw1_tg
	    
	    if $do_eval; then 
	      decode_dir=exp/tri4_mmi_b0.1/decode_eval2000_${iter}.mdl_sw1_tg
	      steps/decode.sh --nj 30 --cmd "$decode_cmd"  \
	        --config conf/decode.config --iter $iter \
	        --transform-dir exp/tri4/decode_eval2000_sw1_tg \
	        $graph_dir data/eval2000 $decode_dir
	    
	   		if $has_fisher; then
	   			steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
		          data/lang_sw1_{tg,fsh_fg} data/eval2000 \
		          exp/tri4_mmi_b0.1/decode_eval2000_${iter}.mdl_sw1_{tg,fsh_fg}
	   		fi
	  	fi
	  ) &
	done
	wait
	
	if $do_train; then
		# Now do fMMI+MMI training
		steps/train_diag_ubm.sh --silence-weight 0.5 --nj 50 --cmd "$train_cmd" \
		  700 data/train_nodup data/lang exp/tri4_ali_nodup exp/tri4_dubm
		
		steps/train_mmi_fmmi.sh --learning-rate 0.005  \
		  --boost 0.1 --cmd "$train_cmd"  \
		  data/train_nodup data/lang exp/tri4_ali_nodup exp/tri4_dubm \
		  exp/tri4_denlats_nodup exp/tri4_fmmi_b0.1
	fi

	for iter in 4 5 6 7 8; do
	  (
	    graph_dir=exp/tri4/graph_sw1_tg
	    if $do_eval; then 
	      decode_dir=exp/tri4_fmmi_b0.1/decode_eval2000_it${iter}_sw1_tg
	      steps/decode_fmmi.sh --nj 30 --cmd "$decode_cmd" --iter $iter \
	        --transform-dir exp/tri4/decode_eval2000_sw1_tg  \
	        --config conf/decode.config $graph_dir data/eval2000 $decode_dir
	        
	      if $has_fisher; then
					steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
	          		data/lang_sw1_{tg,fsh_fg} data/eval2000 \
	          		exp/tri4_fmmi_b0.1/decode_eval2000_it${iter}_sw1_{tg,fsh_fg}
	   		fi
	    fi    
	  ) &
	done
	wait
fi

# this will help find issues with the lexicon.
# steps/cleanup/debug_lexicon.sh --nj 300 --cmd "$train_cmd" data/train_nodev data/lang exp/tri4 data/local/dict/lexicon.txt exp/debug_lexicon


# SGMM system.
if $do_sgmm; then
	local/run_sgmm2.sh --has-fisher $has_fisher 
fi

if $do_dnn; then
	# Karel's DNN recipe on top of fMLLR features
	local/nnet/run_dnn.sh --do_train $do_train --do_eval $do_eval --has-fisher $has_fisher 
fi

if $do_nnet2; then
	# Dan's nnet recipe
	local/nnet2/run_nnet2.sh --do_train $do_train --do_eval $do_eval --has-fisher $has_fisher 
fi

# Dan's nnet recipe with online decoding.
# local/online/run_nnet2_ms.sh --has-fisher $has_fisher

# demonstration script for resegmentation.
# local/run_resegment.sh

# demonstration script for raw-fMLLR.  You should probably ignore this.
# local/run_raw_fmllr.sh

if $do_lstm; then
	# nnet3 LSTM recipe
	local/nnet3/run_lstm.sh
	
	# nnet3 BLSTM recipe
	local/nnet3/run_lstm.sh --affix bidirectional \
	                  --lstm-delay " [-1,1] [-2,2] [-3,3] " \
	                        --label-delay 0 \
	                        --cell-dim 1024 \
	                        --recurrent-projection-dim 128 \
	                        --non-recurrent-projection-dim 128 \
	                        --chunk-left-context 40 \
	                        --chunk-right-context 40
fi

# getting results (see RESULTS file)
# for x in 1 2 3a 3b 4a; do grep 'Percent Total Error' exp/tri$x/decode_eval2000_sw1_tg/score_*/eval2000.ctm.filt.dtl | sort -k5 -g | head -1; done