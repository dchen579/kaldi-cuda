// online2bin/online2-wav-nnet3-latgen-faster.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//           2016  Api.ai (Author: Ilya Platonov)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "cudamatrix/cu-allocator.h"
#include "cudadecoder/cuda-decodable.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"
#include <nvToolsExt.h>
#include <cuda_profiler_api.h>
#include <cuda.h>
#include <chrono>

using namespace kaldi;

void GetDiagnosticsAndPrintOutput(const std::string &utt,
                                  const fst::SymbolTable *word_syms,
                                  const CompactLattice &clat,
                                  int64 *tot_num_frames,
                                  double *tot_like) {
  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    return;
  }
  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);

  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
  num_frames = alignment.size();
  likelihood = -(weight.Value1() + weight.Value2());
  *tot_num_frames += num_frames;
  *tot_like += likelihood;
  KALDI_VLOG(2) << "Likelihood per frame for utterance " << utt << " is "
                << (likelihood / num_frames) << " over " << num_frames
                << " frames.";

  if (word_syms != NULL) {
    std::cerr << utt << ' ';
    for (size_t i = 0; i < words.size(); i++) {
      std::string s = word_syms->Find(words[i]);
      if (s == "")
        KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
      std::cerr << s << ' ';
    }
    std::cerr << std::endl;
  }
}

//using a macro here to avoid a ton of paramaters in a function
//while also being able to reuse this in two spots
#define FinishOneDecode() {                                                           \
  std::string &utt = processed.front().first;                                         \
  std::string &key = processed.front().second;                                        \
  CompactLattice clat;                                                                \
  bool valid;                                                                         \
                                                                                      \
  if(batchedDecoderConfig.determinize_lattice_) {                                     \
    valid=CudaDecoder.GetLattice(key,&clat);                                          \
  } else {                                                                            \
    Lattice lat;                                                                      \
    valid=CudaDecoder.GetRawLattice(key,&lat);                                        \
    ConvertLattice(lat,&clat);                                                        \
  }                                                                                   \
  if(valid) {                                                                         \
    GetDiagnosticsAndPrintOutput(utt, word_syms, clat, &num_frames, &tot_like);       \
    if (write_lattice && key==utt ) { /*only write output on first iteration*/        \
      nvtxRangePushA("Lattice Write");                                                \
      clat_writer.Write(utt, clat);                                                   \
      nvtxRangePop();                                                                 \
    }                                                                                 \
  }                                                                                   \
  CudaDecoder.CloseDecodeHandle(key);                                                 \
  processed.pop();                                                                    \
  if(++current_count==count_per_iteration) { /*this utt is the last in an iter*/      \
    auto finish = std::chrono::high_resolution_clock::now();                          \
    std::chrono::duration<double> total_time = finish-start;                          \
    KALDI_LOG << "Iteration: " << output_iter << " ~Aggregate Total Time: "           \
      << total_time.count() << " Total Audio: " << total_audio*output_iter            \
      << " RealTimeX: " << output_iter*total_audio/total_time.count() << std::endl;   \
    current_count=0;                                                                  \
    output_iter++;                                                                    \
  }                                                                                   \
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
      "Reads in wav file(s) and simulates online decoding with neural nets\n"
      "(nnet3 setup), with optional iVector-based speaker adaptation and\n"
      "optional endpointing.  Note: some configuration values and inputs are\n"
      "set via config files whose filenames are passed as options\n"
      "\n"
      "Usage: online2-wav-nnet3-latgen-faster [options] <nnet3-in> <fst-in> "
      "<wav-rspecifier> <lattice-wspecifier>\n";

    std::string word_syms_rxfilename;

    bool write_lattice = true;
    int num_todo = -1;
    int iterations=1;
    ParseOptions po(usage);
    int pipeline_length=2000; //length of pipeline of outstanding requests, this is independent of queue lengths in decoder

    po.Register("write-lattice",&write_lattice, "Output lattice to a file.  Setting to false is useful when benchmarking.");
    po.Register("word-symbol-table", &word_syms_rxfilename, "Symbol table for words [for debug output]");
    po.Register("file-limit", &num_todo, 
        "Limits the number of files that are processed by this driver.  After N files are processed the remaing files are ignored.  Useful for profiling.");
    po.Register("iterations", &iterations, "Number of times to decode the corpus.  Output will be written only once.");

    //Multi-threaded CPU and batched GPU decoder
    BatchedCudaDecoderConfig batchedDecoderConfig;

    RegisterCuAllocatorOptions(&po);
    batchedDecoderConfig.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      return 1;
    }

    g_cuda_allocator.SetOptions(g_allocator_options);
    CuDevice::Instantiate().SelectGpuId("yes");
    CuDevice::Instantiate().AllowMultithreading();

    ThreadedBatchedCudaDecoder CudaDecoder(batchedDecoderConfig);

    std::string nnet3_rxfilename = po.GetArg(1),
      fst_rxfilename = po.GetArg(2),
      wav_rspecifier = po.GetArg(3),
      clat_wspecifier = po.GetArg(4);

    CompactLatticeWriter clat_writer(clat_wspecifier);

    fst::Fst<fst::StdArc> *decode_fst= fst::ReadFstKaldiGeneric(fst_rxfilename);

    CudaDecoder.Initialize(*decode_fst, nnet3_rxfilename);

    delete decode_fst;

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
          << word_syms_rxfilename;

    int32 num_done = 0, num_err = 0;
    double tot_like = 0.0;
    int64 num_frames = 0;
    double total_audio=0;

    nvtxRangePush("Global Timer");
    auto start = std::chrono::high_resolution_clock::now(); //starting timer here so we can measure throughput without allocation overheads

    int count_per_iteration=0;
    int current_count=0;
    int output_iter=1;

    std::queue<std::pair<std::string,std::string> > processed;
    for (int iter=0;iter<iterations;iter++) {
      SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);

      for (; !wav_reader.Done(); wav_reader.Next()) {
        nvtxRangePushA("Utterance Iteration");

        std::string utt = wav_reader.Key();
        std::string key = utt;
        if (iter>0) { 
          //make key unique for subsequent iterations
          key = key + "-" + std::to_string(iter);
        }
        const WaveData &wave_data = wav_reader.Value();
        
        if(iter==0) {
          //calculating number of utterances per iteration
          count_per_iteration++;  
          //calculating total audio time per iteration
          total_audio+=wave_data.Duration();
        }

        CudaDecoder.OpenDecodeHandle(key,wave_data);
        processed.push(pair<string,string>(utt,key));
        num_done++;

        while (processed.size()>=pipeline_length) {
          FinishOneDecode();
        } //end while 

        nvtxRangePop();
        if (num_todo!=-1 && num_done>=num_todo) break;
      } //end utterance loop

    } //End iterations loop

    while (processed.size()>0) {
      FinishOneDecode();
    } //end while

    KALDI_LOG << "Decoded " << num_done << " utterances, "
      << num_err << " with errors.";
    KALDI_LOG << "Overall likelihood per frame was " << (tot_like / num_frames)
      << " per frame over " << num_frames << " frames.";

    delete word_syms; // will delete if non-NULL.

    clat_writer.Close();

    CudaDecoder.Finalize();  
    cudaDeviceSynchronize();

    auto finish = std::chrono::high_resolution_clock::now();
    nvtxRangePop();

    std::chrono::duration<double> total_time = finish-start;

    KALDI_LOG << "Overall: " << " Aggregate Total Time: " << total_time.count()
      << " Total Audio: " << total_audio 
      << " RealTimeX: " << total_audio/total_time.count() << std::endl;

    return 0;

    //return (num_done != 0 ? 0 : 1);
  } catch (const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()

