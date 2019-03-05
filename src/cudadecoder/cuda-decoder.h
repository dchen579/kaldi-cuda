// cudadecoder/cuda-decoder.h
// TODO nvidia apache2
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

#ifndef KALDI_DECODER_CUDA_DECODER_H_
#define KALDI_DECODER_CUDA_DECODER_H_

#include <cuda_runtime_api.h>
#include <tuple>
#include <vector>

#include "cudadecoder/cuda-decodable-itf.h"
#include "cudadecoder/cuda-decoder-utils.h"
#include "lat/kaldi-lattice.h"
#include "nnet3/decodable-online-looped.h"
#include "util/stl-utils.h"

// A decoder channel is linked to one utterance. Frames
// from the same must be sent to the same channel.
//
// A decoder lane is where the computation actually happens
// a decoder lane is given a frame and its associated channel
// and does the actual computation
//
// An analogy would be lane -> a core, channel -> a software thread

// Number of GPU decoder lanes
#define KALDI_CUDA_DECODER_MAX_N_LANES 200

// If we're at risk of filling the tokens queue,
// the beam is reduced to keep only the best candidates in the
// remaining space
// We then slowly put the beam back to its default value
// beam_next_frame = min(default_beam, RECOVER_RATE * beam_previous_frame)
#define KALDI_CUDA_DECODER_ADAPTIVE_BEAM_RECOVER_RATE 1.2f

// Defines for the cuda decoder kernels
// It shouldn't be necessary to change the DIMX of the kernels

// Below that value, we launch the persistent kernel for NonEmitting
#define KALDI_CUDA_DECODER_NONEM_LT_MAX_NARCS 4096

// We know we will have at least X elements in the hashmap
// We allocate space for X*KALDI_CUDA_DECODER_HASHMAP_CAPACITY_FACTOR elements
// to avoid having too much collisions
#define KALDI_CUDA_DECODER_HASHMAP_CAPACITY_FACTOR 1

// Max size of the total kernel arguments
// 4kb for compute capability >= 2.0
#define KALDI_CUDA_DECODER_MAX_KERNEL_ARGUMENTS_BYTE_SIZE (4096)

// When applying the max-active, we need to compute a topk
// to perform that (soft) topk, we compute a histogram
// here we define the number of bins in that histogram
// it has to be less than the number of 1D threads 
#define KALDI_CUDA_DECODER_HISTO_NBINS 255

// Adaptive beam parameters 
// We will decrease the beam when we detect that we are generating too many tokens
// for the first segment of the aux_q, we don't do anything (keep the original beam)
// the first segment is made of (aux_q capacity)/KALDI_CUDA_DECODER_ADAPTIVE_BEAM_STATIC_SEGMENT
// then we will decrease the beam step by step, until 0.
// we will decrease the beam every m elements, with:
// x = (aux_q capacity)/KALDI_CUDA_DECODER_ADAPTIVE_BEAM_STATIC_SEGMENT (static segment
// y = (aux_q capacity) - x
// m = y / KALDI_CUDA_DECODER_ADAPTIVE_BEAM_NBINS
#define KALDI_CUDA_DECODER_ADAPTIVE_BEAM_STATIC_SEGMENT 4
#define KALDI_CUDA_DECODER_ADAPTIVE_BEAM_NBINS 8

namespace kaldi {
typedef fst::StdArc StdArc;
typedef StdArc::Weight StdWeight;
typedef StdArc::Label Label;
typedef StdArc::StateId StateId;

enum OVERFLOW_TYPE {
  OVERFLOW_NONE = 0,
  OVERFLOW_MAIN_Q = 1,
  OVERFLOW_AUX_Q = 2
};

// Forward declaration 
class DeviceParams;
class KernelParams;
class HashmapValueT; // TODO why forward?

class CudaDecoder;

struct CudaDecoderConfig {
  BaseFloat default_beam;
  BaseFloat lattice_beam;
  int32 max_tokens;
  int32 max_tokens_per_frame;
  int32 nlanes;
  int32 nchannels;
  int32 max_active;

  CudaDecoderConfig()
      : default_beam(15.0),
        lattice_beam(10.0),
        max_tokens(2000000),
        max_tokens_per_frame(1000000),
        max_active(10000) {}

  void Register(OptionsItf *opts) {
    opts->Register(
        "beam", &default_beam,
        "Decoding beam.  Larger->slower, more accurate. The beam may be"
        "decreased if we are generating too many tokens compared to "
        "what the queue can hold (max_tokens_per_frame)");
    opts->Register("max-tokens-pre-allocated", &max_tokens,
                   "Total number of tokens pre-allocated (equivalent to "
                   "reserve in a std vector).  If actual usaged exceeds this "
                   "performance will be degraded");
    opts->Register("max-tokens-per-frame", &max_tokens_per_frame,
                   "Number of tokens allocated per frame. If actual usaged "
                   "exceeds this the results are undefined.");
    opts->Register("lattice-beam", &lattice_beam, "Lattice generation beam");
    opts->Register("max-active", &max_active,
                   "Max number of tokens active for each frame");
  }
  void Check() const {
    KALDI_ASSERT(default_beam > 0.0 && max_tokens > 0 &&
                 max_tokens_per_frame > 0 && lattice_beam >= 0 &&
                 max_active > 1);
  }
};

class CudaDecoder {
 public:
  // Creating a new CudaDecoder, associated to the FST fst
  // nlanes and nchannels are defined as follow

  // A decoder channel is linked to one utterance.
  // When we need to perform decoding on an utterance,
  // we pick an available channel, call InitDecoding on that channel
  // (with that ChannelId in the channels vector in the arguments)
  // then call AdvanceDecoding whenever frames are ready for the decoder
  // for that utterance (also passing the same ChannelId to AdvanceDecoding)
  //
  // A decoder lane is where the computation actually happens
  // a decoder lane is channel, and perform the actual decoding
  // of that channel.
  // If we have 200 lanes, we can compute 200 utterances (channels)
  // at the same time. We need many lanes in parallel to saturate the big GPUs
  //
  // An analogy would be lane -> a CPU core, channel -> a software thread
  // A channel saves the current state of the decoding for a given utterance.
  // It can be kept idle until more frames are ready to be processed
  //
  // We will use as many lanes as necessary to saturate the GPU, but not more.
  // A lane has an higher memory usage than a channel. If you just want to be
  // able to
  // keep more audio channels open at the same time (when I/O is the bottleneck
  // for instance,
  // typically in the context of online decoding), you should instead use more
  // channels.
  //
  // A channel is typically way smaller in term of memory usage, and can be used
  // to oversubsribe lanes in the context of online decoding
  // For instance, we could choose nlanes=200 because it gives us good
  // performance
  // on a given GPU. It gives us an end-to-end performance of 3000 XRTF. We are
  // doing online,
  // so we only get audio at realtime speed for a given utterance/channel.
  // We then decide to receive audio from 2500 audio channels at the same time
  // (each at realtime speed),
  // and as soon as we have frames ready for nlanes=200 channels, we call
  // AdvanceDecoding on those channels
  // In that configuration, we have nlanes=200 (for performance), and
  // nchannels=2500 (to have enough audio
  // available at a given time).
  // Using nlanes=2500 in that configuration would first not be possible (out of
  // memory), but also not necessary.
  // Increasing the number of lanes is only useful if it increases performance.
  // If the GPU is saturated at nlanes=200,
  // you should not increase that number
  CudaDecoder(const CudaFst &fst, const CudaDecoderConfig &config,
              int32 nlanes = 1, int32 nchannels = 1);
  ~CudaDecoder();

  // InitDecoding initializes the decoding, and should only be used if you
  // intend to call AdvanceDecoding() on the channels listed in channels
  void InitDecoding(const std::vector<ChannelId> &channels);

  // AdvanceDecoding on a given batch
  // a batch is defined by the channels vector
  // We can compute N channels at the same time (in the same batch)
  // where N = number of lanes, as defined in the constructor 
  // AdvanceDecoding will compute as many frames as possible while running the full batch
  // when at least one channel has no more frames ready to be computed, AdvanceDecoding returns
  // The user then decides what to do, i.e.:
  // 
  // 1) either remove the empty channel from the channels list
  // and call again AdvanceDecoding
  // 2) or swap the empty channel with another one that has frames ready
  // and call again AdvanceDecoding
  //
  // Solution 2) should always be preferred because we need to run full, big batches to saturate the GPU
  //
  // If max_num_frames is >= 0 it will decode no more than
  // that many frames.
  void AdvanceDecoding(const std::vector<ChannelId> &channels,
                       std::vector<CudaDecodableInterface *> &decodables,
                       int32 max_num_frames = -1);

  // Returns the number of frames already decoded in a given channel
  int32 NumFramesDecoded(ChannelId ichannel) const;

  // GetBestPath gets the decoding traceback. If "use_final_probs" is true
  // AND we reached a final state, it limits itself to final states;
  // otherwise it gets the most likely token not taking int32o account
  // final-probs.
  // fst_out will be empty (Start() == kNoStateId) if nothing was available due
  // to
  // search error.
  // If Decode() returned true, it is safe to assume GetBestPath will return
  // true.
  // It returns true if the output lattice was nonempty (i.e. had states in it);
  // using the return value is deprecated.
  bool GetBestPath(const std::vector<ChannelId> &channels,
                   std::vector<Lattice *> &fst_out_vec,
                   bool use_final_probs = true);
  bool GetBestPath(Lattice *fst_out,
                   bool use_final_probs = true);  // batch size = 1
  bool GetRawLattice(const std::vector<ChannelId> &channels,
                     std::vector<Lattice *> &fst_out_vec, bool use_final_probs);

  // GetBestCost sets in *min the token's best cost in the main_q
  // it also sets in *arg the index of that token (argmin)
  // is isfinal is true, we take into account the final costs
  void GetBestCost(
      const std::vector<ChannelId> &channels, bool isfinal,
      std::vector<std::pair<int32, CostType>> *argmins,
      std::vector<std::vector<std::pair<int, float>>> *list_finals_token_idx,
      std::vector<bool> *has_reached_final);

  //
  // Data structures used by the kernels
  //

  // Count of tokens and arcs in a queue
  // narcs = sum(number of arcs going out of token i next state) for each token
  // in the queue
  // We use this struct to keep the two int32s adjacent in memory
  // we need this in order to update both using an atomic64 operation
  struct TokenAndArcCount {
    int32 ntokens;
    int32 narcs;
  };

  // Union structure of the TokenAndArcCount
  // We use split to access the int32s
  // We use both to update both using an atomic64
  union TokenAndArcCountUnion {
    TokenAndArcCount split;
    unsigned long long both;
  };

  //
  // Used for the cutoff
  // cutoff = min_cost + beam
  // We store both separatly because we have an adaptive beam
  // We may change the beam after discovering min_cost
  // we need to keep track of min_cost to apply the new beam
  // (we don't know what the old beam was)
  //
  // Native float and Integers version
  //
  struct MinCostAndBeam {
    CostType min_cost;
    CostType beam;
  };

  struct MinCostAndBeamIntegers {
    IntegerCostType min_cost;
    IntegerCostType beam;
  };

 private:
  // Computes the initial channel
  // The initial channel is used to initialize a channel
  // when a new utterance starts
  void ComputeInitialChannel();

  // Updates *h_kernel_params using channels
  void SetChannelsInKernelParams(const std::vector<ChannelId> &channels);

  // Called by InitDecoding. Does the part of InitDecoding that needs to be done
  // on the device
  void InitDecodingOnDevice(const std::vector<ChannelId> &channels);
  //
  // ExpandArcs kernel
  // This kernel reads token from the main_q and uses the FST graph
  // to compute new token queue in aux_q
  // To do this, for each token in the main_q, we traverse each arcs going
  // out of that token's next state. If that arc's next state is a valid
  // candidate,
  // we create a new token and add it to aux_q.
  // For more information on the condition for the creation of a new token,
  // please refer to http://kaldi-asr.org/doc/decoders.html
  //
  // main_q_narcs_estimate is used to decide how many threads to launch
  // it is not used inside the kernel, where the exact value will be used
  //

  void ExpandArcs(bool is_emitting, int32 main_q_narcs_estimate);

  //
  // PreprocessAndContract kernel
  //
  // The Preprocess* kernels are used before executing Expand
  // Computing data members needed by Expand (preprocess) and prune tokens on
  // the fly (contract)
  //
  // Pseudo code of the PreprocessAndContract kernel :
  // - For each token in the aux_q,
  //          compute bool is_best = (token.cost < cutoff)
  //                                && (token.cost ==
  //                                d_state_best_costs[token.nextstate])
  // If is_best, then :
  //         1) append this token to the main_q
  //         2) compute out_degree = (# of outgoing arcs from token.nextstate)
  //         3) compute the prefix sum of those out_degrees for all tokens
  //         appended in the main_q
  //         4) save that prefix sum in d_main_q_degrees_prefix_sum
  // Else
  //    Do nothing (this token is pruned)
  //
  // After executing PreprocessAndContract,
  // - aux_q is considered empty. d_aux_q_end was resetted to 0
  // - all tokens generated buy PreprocessAndContract are in the main_q,
  //   in the index range [d_main_q_local_offset, d_main_q_end[
  //
  // Note : Using a trick, we can compute everything (including the prefix sum)
  // using only one kernel (without global syncs)
  // We don't need to call FinalizePreprocessInPlace() after
  // PreprocessAndContract
  //
  // aux_q_size_estimate is used to decide how many threads to launch
  // it is not used inside the kernel, where the exact value will be used
  //

  void PreprocessAndContract(int32 aux_q_size_estimate);

  //
  // PreprocessInPlace kernel
  //
  // The Preprocess* kernels are used before executing Expand
  // Computing data members needed by Expand (preprocess) in place, without
  // modifying queues
  //
  // This is used when the input tokens were already used to generate children
  // tokens
  // In practice, it happens when we are in a ProcessEmitting stage
  // The input tokens in ProcessEmitting at frame i were already used in
  // ProcessNonEmitting at frame (i-1)
  // It means that those input tokens already have children. Those children
  // token refer to the index of their
  // input tokens in their token.prev_token data member.
  // If we were to use PreprocessAndContract,
  // the pruning stage would change the tokens indexes - and we would have to
  // reindex those prev_token,
  // hence the need for a PreprocessInPlace
  //
  // Pseudo code of the PreprocessInPlace kernel :
  // - For each token in the range [local_offset, end[ of the main_q,
  //          compute bool is_best = (token.cost < cutoff)
  //                                && (token.cost ==
  //                                d_state_best_costs[token.nextstate])
  //
  //          compute out_degree = is_best
  //                               ? (# of outgoing arcs from token.nextstate)
  //                               : 0
  //
  //  By artifically setting the out_degree to 0 we tell the expand kernel to
  //  completely ignore that token
  //  (we rely on the 1 arc = 1 thread exact load balancing of the expand
  //  kernel)
  //
  // Then we compute the data needed by the expand kernel :
  //         1) compute the prefix sum of those out_degrees
  //         3) save that prefix sum in d_main_q_degrees_prefix_sum
  //
  // After executing PreprocessInPlace,
  // The "active" main_q[local_offset, end[ range stays the same.
  //
  // Note : Only the first pass of the prefix sum is computed in that kernel. We
  // then need to call
  // ResetStateBestCostLookupAndFinalizePreprocessInPlace after
  // PreprocessInPlace
  //
  // main_q_size_estimate is used to decide how many threads to launch
  // it is not used inside the kernel, where the exact value will be used
  //

  void LoadChannelsStateToLanesCPU();
  void SaveChannelsStateFromLanesCPU();

  //
  // FinalizeProcessNonemitting
  // This kernel is called at the end of the ProcessNonEmitting computation
  // it is used when ProcessNonEmitting generate a small number of new tokens at
  // each iteration
  // to avoid calling heavy-lifting kernels such as ExpandArcs too many times,
  // we instead use
  // FinalizeProcessNonemitting that uses only one CTA
  // By using one CTA, we can sync all threads inside the kernel, and iterate
  // until convergence
  // without lauching new kernels
  // This meta-kernel performs :
  // while we have non-emitting arcs to traverse:
  //      (1) Preprocess and contract
  //      (2) Expand
  // This meta-kernel does not call the PreprocessAndContract or Expand kernels
  // it uses simplified implementations (for one CTA) of those
  //
  void FinalizeProcessNonemitting();

  // InitStateCost initializes all costs to +INF in d_state_best_cost at the
  // beginning of the computation
  void InitStateBestCostLookup();

  // If we have more than max_active_ tokens in the queue, we will compute a new
  // beam,
  // that will only keep max_active_ tokens
  void ApplyMaxActiveAndReduceBeam(bool use_aux_q);
  //
  // This kernel contains both ResetStateCostLookup and FinalizePreprocess in
  // place.
  //
  // ResetStateCostLookup :
  //
  // We need to reset d_state_best_cost between frames. We could use
  // InitStateCost
  // but a large portion of the lookup table has not been used
  // ResetStateBestCostLookupAndFinalizePreprocessInPlace resets only the costs
  // that are not +INF, using the d_main_q_state to do it
  // d_main_q_state contains the list of states that have been considered and
  // have a best cost < +INF
  //
  // FinalizePreprocessInPlace :
  //
  // This kernel is responsible to compute the second pass of the
  // prefix sum. Must be called between PreprocessInPlace and ExpandArcs
  //
  //
  // main_q_size_estimate is used to decide how many threads to launch
  // it is not used inside the kernel, where the exact value will be used
  //
  // CheckOverflow
  // If a kernel sets the flag h_q_overflow, we send a warning to stderr
  void CheckOverflow();

  // Evaluates func for each lane, returning the max of all return values
  int32 GetMaxForAllLanes(std::function<int32(const LaneCounters &)> func);

  // Copy the lane counters back to host, async, using stream st
  void CopyLaneCountersToHostAsync(cudaStream_t st);

  template <typename T>
  void PerformConcatenatedCopy(std::function<int32(const LaneCounters &)> func,
                               LaneMatrixInterface<T> src, T *d_concat,
                               T *h_concat, cudaStream_t st,
                               std::vector<int32> *lanes_offsets_ptr);
  template <typename T>
  void MoveConcatenatedCopyToVector(const std::vector<int32> &lanes_offsets,
                                    T *h_concat,
                                    std::vector<std::vector<T>> *vecvec);
  //
  // Data members
  //
  // Pointers in h_* refer to data on the CPU memory
  // Pointers in d_* refer to data on the GPU memory

  // The CudaFst data structure contains the FST graph
  // in the CSR format
  const CudaFst fst_;

  //
  // Tokens queues
  //
  // We have two token queues :
  // - the main queue
  // - the auxiliary queue
  //
  // The auxiliary queue is used to store the raw output of ExpandArcs.
  // We then prune that aux queue and move the survival tokens in the main
  // queue.
  // Tokens stored in the main q can then be used to generate new tokens (using
  // ExpandArcs)
  //
  // As a reminder, here's the data structure of a token :
  //
  // struct Token { state, cost, prev_token, arc_idx }
  //
  // For performance reasons, we split the tokens in three parts :
  // { state } , { cost }, { prev_token, arc_idx }
  // Each part has its associated queue
  // For instance, d_main_q_state[i], d_main_q_cost[i], d_main_q_info[i]
  // all refer to the same token (at index i)
  // The data structure InfoToken contains { prev_token, arc_idx }
  //
  // Note : We cannot use the aux queue to generate new tokens
  // (ie we cannot use the aux queue as an input of ExpandArcs)
  // The generated tokens would have parents in the aux queue,
  // identifying them using their indexes in the queue. Those indexes
  // are not finals because the aux queue will be pruned.
  //

  //
  // Parameters used by a decoder lane
  // At the end of each frame, we know everything stored
  // by those parameters (lookup table, aux_q, etc.)
  // is back to its original state
  // We can reuse it for another frame/channel without doing anything
  //
  LaneCounters *h_lanes_counters_;
  ChannelCounters *h_channels_counters_;
  int32 nlanes_, nchannels_;

  // Contain the various counters used by lanes/channels, such as main_q_end,
  // main_q_narcs..
  DeviceChannelMatrix<ChannelCounters> d_channels_counters_;
  DeviceLaneMatrix<LaneCounters> d_lanes_counters_;

  // main_q_* TODO comments
  DeviceChannelMatrix<int2> d_main_q_state_and_cost_;
  DeviceLaneMatrix<float2> d_main_q_extra_cost_;
  DeviceLaneMatrix<CostType> d_main_q_acoustic_cost_;

  // d_main_q_info_ is only needed as a buffer when creating the
  // tokens. It is not needed by the next frame computation
  // We send it back to the host, and at the end of a frame's computation,
  // it can be used by another channel. That's why it's in
  // "LaneParams", and not "ChannelParams"
  DeviceLaneMatrix<InfoToken> d_main_q_info_;

  // Same thing for the aux q
  DeviceLaneMatrix<int2> d_aux_q_state_and_cost_;  // TODO int_cost
  DeviceLaneMatrix<CostType> d_aux_q_acoustic_cost_;
  DeviceLaneMatrix<InfoToken> d_aux_q_info_;

  DeviceLaneMatrix<int32> d_histograms_;

  // The load balancing of the Expand kernel relies on the prefix sum of the
  // degrees
  // of the state in the queue (more info in the ExpandKernel implementation)
  // That array contains that prefix sum. It is set by the "Preprocess*" kernels
  // and used by the Expand kernel
  DeviceChannelMatrix<int32> d_main_q_degrees_prefix_sum_;
  DeviceLaneMatrix<int32> d_main_q_extra_prev_tokens_prefix_sum_;
  DeviceLaneMatrix<int32> d_main_q_representative_id_;
  DeviceLaneMatrix<int32> d_main_q_n_extra_prev_tokens_local_idx_;
  DeviceLaneMatrix<InfoToken> d_main_q_extra_prev_tokens_;

  // When generating d_main_q_degrees_prefix_sum we may need to do it in three
  // steps
  // (1) First generate the prefix sum inside each CUDA blocks
  // (2) then generate the prefix sum of the sums of each CUDA blocks
  // (3) Use (1) and (2) to generate the global prefix sum
  // Data from step 1 and 3 is stored in d_main_q_degrees_prefix_sum
  // Data from step 2 is stored in d_main_q_degrees_block_sums_prefix_sum
  // Note : this is only used by PreprocessInPlace
  // PreprocessAndContract uses a trick to compute the global prefix sum in one
  // pass
  DeviceLaneMatrix<int2> d_main_q_block_sums_prefix_sum_;

  // d_main_q_arc_offsets[i] = fst_.arc_offsets[d_main_q_state[i]]
  // we pay the price for the random memory accesses of fst_.arc_offsets in the
  // preprocess kernel
  // we cache the results in d_main_q_arc_offsets which will be read in a
  // coalesced fashion in expand
  DeviceChannelMatrix<int32> d_main_q_arc_offsets_;

  std::vector<BaseFloat *> h_loglikehoods_ptrs_;
  DeviceLaneMatrix<HashmapValueT> d_hashmap_values_;

  DeviceParams *h_device_params_;
  KernelParams *h_kernel_params_;

  // When starting a new utterance,
  // init_channel_id is used to initialize a channel
  int32 init_channel_id_;

  // is_channel_busy[i] <=> channel i is currently
  // being used by a decoder lane
  // TODO std::bitset is_channel_busy;

  // CUDA streams
  // kernels are launched in compute_st
  // copies in copy_st
  // we use two streams to overlap copies and kernels
  // we synchronize the two using events
  cudaStream_t compute_st_, copy_st_;

  // CUDA events

  // We need to synchronize the streams copy_st and compute_st
  // because of data dependency : they both have to read or write to the main_q
  // when we're done copying the old main_q to the CPU, we trigger
  // can_write_to_main_q
  cudaEvent_t can_write_to_main_q_;

  // At the end of Preprocess kernels we set h_main_q_narcs (pinned memory)
  // this event is set in the pipeline after Preprocess kernels to inform that
  // data is ready to be read
  cudaEvent_t can_read_h_main_q_narcs_;

  //
  // This kernel is triggered when finalize non emitting is about to start
  //
  cudaEvent_t before_finalize_nonemitting_kernel_;

  // h_main_q_end is final for this frame
  // triggered at the end of a frame computation
  cudaEvent_t can_read_final_h_main_q_end_;

  cudaEvent_t can_use_acoustic_cost_;
  cudaEvent_t can_use_infotoken_;
  cudaEvent_t can_use_extra_cost_;

  // When we generate a new tokens list we only keep candidates
  // that have a cost < best_cost_in_the_queue + beam
  // At first beam = default_beam_
  // We may decrease that beam if we are generating too many tokens
  // (adaptive beam)
  CostType default_beam_;
  CostType lattice_beam_;
  bool generate_lattices_;

  int32 max_tokens_;
  int32 max_active_;
  int32 max_tokens_per_frame_;
  int32 hashmap_capacity_;

  // Keep track of the number of frames decoded in the current file.
  std::vector<int32> num_frames_decoded_;
  std::vector<std::vector<int32>> frame_offsets_;

  // Used when generate_lattices
  std::vector<std::vector<InfoToken>> h_all_tokens_info_;
  std::vector<std::vector<CostType>> h_all_tokens_acoustic_cost_;
  std::vector<std::vector<InfoToken>> h_all_tokens_extra_prev_tokens_;
  std::vector<std::vector<float2>> h_all_tokens_extra_prev_tokens_extra_cost_;

  float2 *h_extra_cost_concat_, *d_extra_cost_concat_;
  InfoToken *h_infotoken_concat_, *d_infotoken_concat_;
  CostType *h_acoustic_cost_concat_, *d_acoustic_cost_concat_;
  InfoToken *h_extra_prev_tokens_concat_;

  std::vector<int32> h_main_q_end_lane_offsets_,
      h_emitting_main_q_end_lane_offsets_;
  std::vector<int32> h_n_extra_prev_tokens_lane_offsets_;
  // Used for debugging purposes
  // only malloc'ed if necessary
  int32 *h_debug_buf1_, *h_debug_buf2_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(CudaDecoder);
};

}  // end namespace kaldi.

#endif
