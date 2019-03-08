// cudadecoder/cuda-decoder-utils.h
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun, Justin Luitjens, Ryan Leary
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_DECODER_CUDA_DECODER_UTILS_H_
#define KALDI_DECODER_CUDA_DECODER_UTILS_H_
#include "cudamatrix/cu-device.h"
#include "util/stl-utils.h"

#define KALDI_CUDA_DECODER_DIV_ROUND_UP(a, b) ((a + b - 1) / b)

#define KALDI_CUDA_DECODER_ASSERT(val, recoverable)                     \
  {                                                                     \
    if ((val) != true) {                                                \
      throw CudaDecoderException("KALDI_CUDA_DECODER_ASSERT", __FILE__, \
                                 __LINE__, recoverable)                 \
    }                                                                   \
  }
// Macro for checking cuda errors following a cuda launch or api call
#define KALDI_DECODER_CUDA_CHECK_ERROR()                                  \
  {                                                                       \
    cudaError_t e = cudaGetLastError();                                   \
    if (e != cudaSuccess) {                                               \
      throw CudaDecoderException(cudaGetErrorName(e), __FILE__, __LINE__, \
                                 false);                                  \
    }                                                                     \
  }

#define KALDI_DECODER_CUDA_API_CHECK_ERROR(e)                             \
  {                                                                       \
    if (e != cudaSuccess) {                                               \
      throw CudaDecoderException(cudaGetErrorName(e), __FILE__, __LINE__, \
                                 false);                                  \
    }                                                                     \
  }

#define KALDI_CUDA_DECODER_1D_KERNEL_LOOP(i, n)                \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define KALDI_CUDA_DECODER_1D_BLOCK_OFFSET_KERNEL_LOOP(offset, th_idx, n) \
  for (int offset = blockIdx.x * blockDim.x, th_idx = threadIdx.x;        \
       offset < (n); offset += blockDim.x * gridDim.x)

#define KALDI_CUDA_DECODER_IS_LAST_1D_THREAD() (threadIdx.x == (blockDim.x - 1))

#define KALDI_CUDA_DECODER_BATCH_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.y; i < (n); i += gridDim.y)

#define KALDI_CUDA_DECODER_DIV_ROUND_UP(a, b) ((a + b - 1) / b)

#define KALDI_CUDA_DECODER_1D_BLOCK 256
#define KALDI_CUDA_DECODER_LARGEST_1D_BLOCK 1024
#define KALDI_CUDA_DECODER_ONE_THREAD_BLOCK 1

namespace kaldi {
namespace CudaDecode {

// Returning the number of CTAs to launch for (N,M) elements to compute
// M is usually the batch size
inline dim3 KALDI_CUDA_DECODER_NUM_BLOCKS(int N, int M) {
  dim3 grid;
  // TODO MAX_NUM_BLOCKS.
  grid.x = KALDI_CUDA_DECODER_DIV_ROUND_UP(N, KALDI_CUDA_DECODER_1D_BLOCK);
  grid.y = M;
  return grid;
}

typedef float CostType;
// IntegerCostType is the type used in the lookup table d_state_best_cost
// and the d_cutoff
// We use a 1:1 conversion between CostType <--> IntegerCostType
// IntegerCostType is used because it triggers native atomic operations
// (CostType does not)
typedef int32 IntegerCostType;
typedef int32 LaneId;
typedef int32 ChannelId;

// On the device we compute everything by batch
// Data is stored as 2D matrices (BatchSize, 1D_Size)
// For example, for the token queue, (BatchSize, max_tokens_per_frame_)
// DeviceMatrix owns the data but is not used to access it.
// DeviceMatrix is inherited in DeviceLaneMatrix and DeviceChannelMatrix
// those two classes do the same thing, except that they belong either to a
// channel or lane
// that inheritance is done to clarify the code and help debugging
//
// To actually access the data, we should request an interface through
// GetInterface
// That interface contains both host and cuda code to access the data.
template <typename T>
// if necessary, make a version that always use ld_ as the next power of 2
class DeviceMatrix {
  T *data_;
  void Allocate() {
    KALDI_ASSERT(nrows_ > 0);
    KALDI_ASSERT(ld_ > 0);
    KALDI_ASSERT(!data_);
    data_ = static_cast<T *>(
        CuDevice::Instantiate().Malloc((size_t)nrows_ * ld_ * sizeof(*data_)));
    KALDI_ASSERT(data_);
  }
  void Free() {
    KALDI_ASSERT(data_);
    CuDevice::Instantiate().Free(data_);
  }

 protected:
  int32 ld_;     // leading dimension
  int32 nrows_;  // leading dimension
 public:
  DeviceMatrix() : data_(NULL), ld_(0), nrows_(0) {}

  virtual ~DeviceMatrix() {
    if (data_) Free();
  }

  void Resize(int32 nrows, int32 ld) {
    KALDI_ASSERT(nrows > 0);
    KALDI_ASSERT(ld > 0);
    nrows_ = nrows;
    ld_ = ld;
  }

  T *MutableData() {
    if (!data_) Allocate();
    return data_;
  }
  // abstract getInterface...
};

// The interfaces contains CUDA code
// We will declare them in a .cu file
// Only doing a forward declaration for now
template <typename T>
class LaneMatrixInterface;
template <typename T>
class ChannelMatrixInterface;
template <typename T>

class DeviceLaneMatrix : public DeviceMatrix<T> {
 public:
  LaneMatrixInterface<T> GetInterface() {
    return {this->MutableData(), this->ld_};
  }

  T *lane(const int32 ilane) { return &this->MutableData()[ilane * this->ld_]; }
};

template <typename T>
class DeviceChannelMatrix : public DeviceMatrix<T> {
 public:
  ChannelMatrixInterface<T> GetInterface() {
    return {this->MutableData(), this->ld_};
  }
  T *channel(const int32 ichannel) {
    return &this->MutableData()[ichannel * this->ld_];
  }
};

// LaneCounters/ChannelCounters
// The counters are all the singular values associated to a lane/channel
// For instance  the main queue size. Or the min_cost of all tokens in that
// queue
// LaneCounters are used during computation
struct LaneCounters {
  // Contains both main_q_end and narcs
  // End index of the main queue
  // only tokens at index i with i < main_q_end
  // are valid tokens
  // Each valid token the subqueue main_q[main_q_local_offset, main_q_end[ has
  // a number of outgoing arcs (out-degree)
  // main_q_narcs is the sum of those numbers
  // We sometime need to update both end and narcs at the same time using a
  // single atomic,
  // which is why they're packed together
  int2 main_q_narcs_and_end;
  // contains the requested queue length which can
  // be larger then the actual queue length in the case of overflow
  int32 main_q_requested;
  int32 aux_q_requested;
  int32 aux_q_end;
  int32 post_expand_aux_q_end;  // used for double buffering
  // Some tokens in the same frame share the same token.next_state
  // main_q_n_extra_prev_tokens is the count of those tokens
  int32 main_q_n_extra_prev_tokens;
  // Depending on the value of the parameter "max_tokens_per_frame"
  // we can end up with an overflow when generating the tokens for a frame
  // We try to prevent this from happening using an adaptive beam
  // If an overflow happens, then the kernels no longer insert any data into
  // the queues and set overflow flag to true.
  // queue length.
  // Even if that flag is set, we can continue the execution (quality
  // of the output can be lowered)
  // We use that flag to display a warning to the user
  int32 q_overflow;
  // ExpandArcs reads the tokens in the index range [main_q_local_offset, end[
  int32 main_q_local_offset;
  // We transfer the tokens back to the host at the end of each frame.
  // Which means that tokens at a frame  n > 0 have an offset compared to to
  // those
  // in frame n-1. main_q_global_offset is the overall offset of the current
  // main_q,
  // since frame 0
  // It is used to set the prev_token index.
  int32 main_q_global_offset;
  // Same thing, but for main_q_n_extra_prev_tokens (those are also transfered
  // back to host)
  int32 main_q_extra_prev_tokens_global_offset;

  // Minimum token for that frame
  IntegerCostType min_int_cost;
  // Current beam. Can be different from default_beam,
  // because of the AdaptiveBeam process, or because of
  // ApplyMaxActiveAndReduceBeam
  IntegerCostType int_beam;
  // Adaptive beam. The validity says until which index this adaptive beam is
  // valid.
  // After that index, we need to lower the adaptive beam
  int2 adaptive_int_beam_with_validity_index;

  // min_cost + beam
  IntegerCostType int_cutoff;

  // --- Only valid after calling GetBestCost
  // min_cost and its arg. Can be different than min_cost, because we may
  // include final costs
  int2 min_int_cost_and_arg;
  // Number of final tokens with cost < best + lattice_beam
  int32 nfinals;
  int32 has_reached_final;  // if there's at least one final token in the queue
};

// Channel counters
// Their job is to save the state of a channel, when this channel is idle
// The channel counters are loaded into the lane counters during the context
// switches
struct ChannelCounters {
  // All the following values are just saved values from LaneCounters
  // from the latest context-switch
  int2 prev_main_q_narcs_and_end;
  int32 prev_main_q_n_extra_prev_tokens;
  int32 prev_main_q_global_offset;
  int32 prev_main_q_extra_prev_tokens_global_offset;
  CostType prev_beam;

  // Only valid after calling GetBestCost
  // different than min_int_cost : we include the "final" cost
  int2 min_int_cost_and_arg_with_final;
  int2 min_int_cost_and_arg_without_final;
  //
};

class CudaDecoderException : public std::exception {
 public:
  CudaDecoderException(const char *str_, const char *file_, int line_,
                       const bool recoverable_)
      : str(str_),
        file(file_),
        line(line_),
        buffer(std::string(file) + ":" + std::to_string(line) + " :" +
               std::string(str)),
        recoverable(recoverable_) {}
  const char *what() const throw() { return buffer.c_str(); }

  const char *str;
  const char *file;
  const int line;
  const std::string buffer;
  const bool recoverable;
};

// InfoToken contains data that needs to be saved for the backtrack
// in GetBestPath/GetRawLattice
// We don't need the token.cost or token.next_state.
struct __align__(8) InfoToken {
  int32 prev_token;
  int32 arc_idx;
  bool IsUniqueTokenForStateAndFrame() {
    // This is a trick used to save space and PCI-E bandwidth (cf
    // preprocess_in_place kernel)
    // This token is associated with a next_state s, created during the
    // processing of frame f.
    // If we have multiple tokens associated with the state s in the frame f,
    // arc_idx < 0 and -arc_idx is the
    // count of such tokens. We will then have to look at another list to read
    // the actually arc_idx and prev_token values
    // If the current token is the only one, prev_token and arc_idx are valid
    // and can be used directly
    return (arc_idx >= 0);
  }

  // Called if this token is linked to others tokens in the same frame (cf
  // comments for IsUniqueTokenForStateAndFrame)
  // return the {offset,size} pair necessary to list those tokens in the
  // extra_prev_tokens list
  // They are stored at offset "offset", and we have "size" of those
  std::pair<int32, int32> GetNextStateTokensList() {
    KALDI_ASSERT(!IsUniqueTokenForStateAndFrame());

    return {prev_token, -arc_idx};
  }
};

__inline__ int32 floatToOrderedIntHost(float floatVal) {
  int32 intVal;
  // Should be optimized away by compiler
  memcpy(&intVal, &floatVal, sizeof(float));
  return (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
}

__inline__ float orderedIntToFloatHost(int32 intVal) {
  intVal = (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
  float floatVal;
  // Should be optimized away by compiler
  memcpy(&floatVal, &intVal, sizeof(float));
  return floatVal;
}

// Hashmap value. Used when computing the hashmap in PostProcessingMainQueue
struct __align__(16) HashmapValueT {
  // Map key : fst state
  int key;
  // Number of tokens associated to that state
  int count;
  // minimum cost for that state + argmin
  int2 min_and_argmin_int_cost;
};

enum OVERFLOW_TYPE {
  OVERFLOW_NONE = 0,
  OVERFLOW_MAIN_Q = 1,
  OVERFLOW_AUX_Q = 2
};

enum QUEUE_ID { MAIN_Q = 0, AUX_Q = 1 };

}  // end namespace CudaDecode
}  // end namespace kaldi

#endif
