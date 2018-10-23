// nnet3/convolution-cudnn.cc

// Copyright      2018  Daniel Galvez
//                2018  Johns Hopkins University (author: Daniel Povey)

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

#include "nnet3/convolution-cudnn.h"

namespace kaldi {
namespace nnet3 {
namespace cudnn_convolution {


namespace {
// Note: anonymous namespaces are now preferred (by the C++ standard) over
// static variables.
const BaseFloat ONE(1.0);
const BaseFloat ZERO(0.0);
}


void ConvolutionComputationConfig::Check() {
  KALDI_ASSERT(num_images > 0 && num_channels_out > 0 &&
               num_channels_in > 0 && filter_height > 0 && filter_width > 0);
  KALDI_ASSERT(filter_stride_vertical > 0 && filter_stride_horizontal > 0 &&
               filter_dilation_vertical > 0 && filter_dilation_horizontal > 0);
  KALDI_ASSERT(input_image_height > 0 && input_image_width > 0 &&
               zero_padding_vertical >= 0 && zero_padding_horizontal >= 0);
}

void ConvolutionComputationConfig::ComputeOutputImageSize() {
  { // This blocks deals with the vertical direction.

    // 'filter_height_reduction' is the amount by which the height of the filter patch
    // reduces the effective height of the input image.  It's the distance between
    // the first and last pixels of the filter patch.  E.g. in a 3x3 kernel it
    // would be 2.
    int32 filter_height_reduction = (filter_height - 1) * filter_dilation_vertical;
    // 'modified_input_height' is the number of times we can shift the filter patch
    // (not yet taking account of any filter stride).  It's a kind of augmented input-image
    // height, after applying zero-padding and subtracting filter_height_reduction.
    int32 modified_input_height =
        input_image_height - filter_height_reduction + (zero_padding_vertical * 2),
        s = filter_stride_vertical;

    // output_image_height equals reduced_input_height divided by s (but rounding
    // up), which is the number of times we can shift the filter patch by
    // filter_stride_vertical_.
    output_image_height = (modified_input_height + s - 1) / s;
  }

  { // This blocks deals with the horizontal direction.

    // 'filter_width_reduction' is the amount by which the width of the filter patch
    // reduces the effective width of the input image.  It's the distance between
    // the first and last pixels of the filter patch.  E.g. in a 3x3 kernel it
    // would be 2.
    int32 filter_width_reduction = (filter_width - 1) * filter_dilation_horizontal;
    // 'modified_input_width' is the number of times we can shift the filter patch
    // (not yet taking account of any filter stride).  It's a kind of augmented input-image
    // width, after applying zero-padding and subtracting filter_width_reduction.
    int32 modified_input_width =
        input_image_width - filter_width_reduction + (zero_padding_horizontal * 2),
        s = filter_stride_horizontal;

    // output_image_width equals reduced_input_width divided by s (but rounding
    // up), which is the number of times we can shift the filter patch by
    // filter_stride_horizontal_.
    output_image_width = (modified_input_width + s - 1) / s;
  }
}

void ConvolutionComputationConfig::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<ConvolutionComputationConfig>");
  WriteBasicType(os, binary, num_images);
  WriteToken(os, binary, "<Channels>");
  WriteBasicType(os, binary, num_channels_in);
  WriteBasicType(os, binary, num_channels_out);
  WriteToken(os, binary, "<Filters>");
  WriteBasicType(os, binary, filter_height);
  WriteBasicType(os, binary, filter_width);
  WriteBasicType(os, binary, filter_stride_vertical);
  WriteBasicType(os, binary, filter_stride_horizontal);
  WriteBasicType(os, binary, filter_dilation_vertical);
  WriteBasicType(os, binary, filter_dilation_horizontal);
  WriteToken(os, binary, "<Input>");
  WriteBasicType(os, binary, input_image_height);
  WriteBasicType(os, binary, input_image_width);
  WriteToken(os, binary, "<Padding>");
  WriteBasicType(os, binary, zero_padding_vertical);
  WriteBasicType(os, binary, zero_padding_horizontal);
  WriteToken(os, binary, "</ConvolutionComputationConfig>");
}

void ConvolutionComputationConfig::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<ConvolutionComputationConfig>");
  ReadBasicType(is, binary, &num_images);
  ExpectToken(is, binary, "<Channels>");
  ReadBasicType(is, binary, &num_channels_in);
  ReadBasicType(is, binary, &num_channels_out);
  ExpectToken(is, binary, "<Filters>");
  ReadBasicType(is, binary, &filter_height);
  ReadBasicType(is, binary, &filter_width);
  ReadBasicType(is, binary, &filter_stride_vertical);
  ReadBasicType(is, binary, &filter_stride_horizontal);
  ReadBasicType(is, binary, &filter_dilation_vertical);
  ReadBasicType(is, binary, &filter_dilation_horizontal);
  ExpectToken(is, binary, "<Input>");
  ReadBasicType(is, binary, &input_image_height);
  ReadBasicType(is, binary, &input_image_width);
  ExpectToken(is, binary, "<Padding>");
  ReadBasicType(is, binary, &zero_padding_vertical);
  ReadBasicType(is, binary, &zero_padding_horizontal);
  ExpectToken(is, binary, "</ConvolutionComputationConfig>");
  ComputeOutputImageSize();
}


ConvolutionComputation::ConvolutionComputation(
    const ConvolutionComputationConfig &config): config_(config) {
  config_.Check();
  config_.ComputeOutputImageSize();
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    InitCudnn();
  }
#endif
}

ConvolutionComputation::ConvolutionComputation() {
#if HAVE_CUDA == 1
  descriptors_initialized_ = false;
#endif
}


#if HAVE_CUDA == 1

#if (KALDI_DOUBLEPRECISION != 0)
#define CUDNN_DATA_BASEFLOAT CUDNN_DATA_DOUBLE
#else
#define CUDNN_DATA_BASEFLOAT CUDNN_DATA_FLOAT
#endif

void ConvolutionComputation::InitCudnn() {
  descriptors_initialized_ = true;

  const ConvolutionComputationConfig &c = config_;

  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&input_desc_));
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&output_desc_));
  CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&params_desc_));
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&bias_desc_));
  CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_));
  CUDNN_SAFE_CALL(cudnnCreateActivationDescriptor(&activation_desc_));

  // Caution: in the following call, the 'height' and 'width' are swapped
  // relative to what the CUDNN interface specifies; this is because Kaldi's
  // notion of what is height vs. width is opposite to CUDNN's.  (There
  // are good reasons for this).
  CUDNN_SAFE_CALL(
      cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NHWC,
                                 CUDNN_DATA_BASEFLOAT, c.num_images,
                                 c.num_channels_in, c.input_image_width,
                                 c.input_image_height));
  // Again: width and height are swapped.
  CUDNN_SAFE_CALL(
      cudnnSetConvolution2dDescriptor(
          conv_desc_,
          c.zero_padding_horizontal, c.zero_padding_vertical,
          c.filter_stride_horizontal, c.filter_stride_vertical,
          c.filter_dilation_horizontal, c.filter_dilation_vertical,
          CUDNN_CROSS_CORRELATION, // TODO: Double check this!
          CUDNN_DATA_BASEFLOAT));

  // Set dimensions of the filters (linear parameters).
  // Again: width and height are swapped.  Per the CUDNN documentation at
  // https://docs.nvidia.com/deeplearning/sdk/pdf/cuDNN-Developer-Guide.pdf for
  // cudnnSetFilter4dDescriptor, setting CUDNN_TENSOR_NHWC as the layout
  // corresponds to KSRC, meaning: num-channels-out, height, width, num-channels-in,
  // where 'height' and 'width' are the filter height and width respectively (e.g. 3
  // and 3 for a 3x3 patch); and these are swapped w.r.t. Kaldi's notion of height and
  // width, so as far as Kaldi is concerned, the strides are, from largest to
  // smallest: num-channels-out, width, height, num-channels-in.
  CUDNN_SAFE_CALL(
      cudnnSetFilter4dDescriptor(params_desc_, CUDNN_DATA_BASEFLOAT,
                                 CUDNN_TENSOR_NHWC, c.num_channels_out,
                                 c.num_channels_in, c.filter_width,
                                 c.filter_height));

  int32 kaldi_width_cudnn_height, kaldi_height_cudnn_width, unused;
  CUDNN_SAFE_CALL(
      cudnnGetConvolution2dForwardOutputDim(conv_desc_, input_desc_,
                                            params_desc_,
                                            &unused, &unused,
                                            &kaldi_width_cudnn_height,
                                            &kaldi_height_cudnn_width));

  if (kaldi_height_cudnn_width != c.output_image_height)
    KALDI_ERR << "Code error: the height from CUDNN " << kaldi_height_cudnn_width
              << " does not match our value " << c.output_image_height;
  if (kaldi_width_cudnn_height != c.output_image_width)
    KALDI_ERR << "Code error: the width from CUDNN " << kaldi_width_cudnn_height
              << " does not match our value " << c.output_image_width;

  // These two member functions depend only on input_desc_,
  // conv_desc_, and params_desc_, so they are safe to call now.
  CUDNN_SAFE_CALL(
      cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NHWC,
                                 CUDNN_DATA_BASEFLOAT, c.num_images,
                                 c.num_channels_out, kaldi_width_cudnn_height,
                                 kaldi_height_cudnn_width));

  // We pad the bias with leading dims of 1, since CUDNN's tensors appear to
  // need a dimension of at least 3.
  int bias_dims[3] = {1, 1, c.num_channels_out};
  int bias_stride[3] = {c.num_channels_out, c.num_channels_out, 1};
  CUDNN_SAFE_CALL(
      cudnnSetTensorNdDescriptor(bias_desc_, CUDNN_DATA_BASEFLOAT, 3,
                                 bias_dims, bias_stride));

  const double DONT_CARE = 0;
  CUDNN_SAFE_CALL(
      cudnnSetActivationDescriptor(activation_desc_, CUDNN_ACTIVATION_IDENTITY,
                                   CUDNN_PROPAGATE_NAN, DONT_CARE));

  int32 requested_algo_count, returned_algo_count;
  CUDNN_SAFE_CALL(cudnnGetConvolutionForwardAlgorithmMaxCount(
      CuDevice::Instantiate().GetCudnnHandle(), &requested_algo_count));

  cudnnConvolutionFwdAlgoPerf_t *forward_results =
      new cudnnConvolutionFwdAlgoPerf_t[requested_algo_count];
  CUDNN_SAFE_CALL(cudnnFindConvolutionForwardAlgorithm(
      CuDevice::Instantiate().GetCudnnHandle(),
      input_desc_,
      params_desc_,
      conv_desc_,
      output_desc_,
      requested_algo_count,
      &returned_algo_count,
      forward_results));

  KALDI_ASSERT(returned_algo_count > 0 &&
               "No algorithms were returned by CUDNN.");
  const cudnnConvolutionFwdAlgoPerf_t& best_forward = forward_results[0];
  KALDI_ASSERT(best_forward.status == CUDNN_STATUS_SUCCESS);
  fwd_algo_ = best_forward.algo;
  delete [] forward_results;

  CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
      CuDevice::Instantiate().GetCudnnHandle(), &requested_algo_count));
  cudnnConvolutionBwdFilterAlgoPerf_t *backward_filter_results =
      new cudnnConvolutionBwdFilterAlgoPerf_t[requested_algo_count];
  CUDNN_SAFE_CALL(cudnnFindConvolutionBackwardFilterAlgorithm(
      CuDevice::Instantiate().GetCudnnHandle(),
      input_desc_,
      output_desc_,
      conv_desc_,
      params_desc_,
      requested_algo_count,
      &returned_algo_count,
      backward_filter_results));
  KALDI_ASSERT(returned_algo_count > 0 &&
               "No algorithms were returned by CUDNN.");
  const cudnnConvolutionBwdFilterAlgoPerf_t& best_backward_filter =
      backward_filter_results[0];
  KALDI_ASSERT(best_backward_filter.status == CUDNN_STATUS_SUCCESS);
  bwd_filter_algo_ = best_backward_filter.algo;
  delete [] backward_filter_results;

  CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
      CuDevice::Instantiate().GetCudnnHandle(), &requested_algo_count));
  cudnnConvolutionBwdDataAlgoPerf_t *backward_data_results =
      new cudnnConvolutionBwdDataAlgoPerf_t[requested_algo_count];
  CUDNN_SAFE_CALL(cudnnFindConvolutionBackwardDataAlgorithm(
      CuDevice::Instantiate().GetCudnnHandle(),
      params_desc_,
      output_desc_,
      conv_desc_,
      input_desc_,
      requested_algo_count,
      &returned_algo_count,
      backward_data_results));
  KALDI_ASSERT(returned_algo_count > 0 &&
               "No algorithms were returned by CUDNN.");
  const cudnnConvolutionBwdDataAlgoPerf_t& best_backward_data =
      backward_data_results[0];
  KALDI_ASSERT(best_backward_data.status == CUDNN_STATUS_SUCCESS);
  bwd_data_algo_ = best_backward_data.algo;
  delete [] backward_data_results;

  ComputeTempSpaceSizes();
}
#endif

#if HAVE_CUDA == 1
void ConvolutionComputation::ComputeTempSpaceSizes() {
  CUDNN_SAFE_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      CuDevice::Instantiate().GetCudnnHandle(),
      input_desc_,
      params_desc_,
      conv_desc_,
      output_desc_,
      fwd_algo_,
      &temp_space_required_forward_));

  CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
      CuDevice::Instantiate().GetCudnnHandle(),
      params_desc_,
      output_desc_,
      conv_desc_,
      input_desc_,
      bwd_data_algo_,
      &temp_space_required_backward_data_));

  CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      CuDevice::Instantiate().GetCudnnHandle(),
      input_desc_,
      output_desc_,
      conv_desc_,
      params_desc_,
      bwd_filter_algo_,
      &temp_space_required_backward_filter_));
}
#endif

#if HAVE_CUDA == 1
void ConvolutionComputation::DestroyCudnn() {
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(input_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(output_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyFilterDescriptor(params_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(bias_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyConvolutionDescriptor(conv_desc_));
  CUDNN_SAFE_CALL(cudnnDestroyActivationDescriptor(activation_desc_));
}
#endif

ConvolutionComputation::~ConvolutionComputation() {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled() && descriptors_initialized_)
    DestroyCudnn();
#endif
}


void ConvolutionComputation::Read(std::istream &is, bool binary) {
  config_.Read(is, binary);
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
    InitCudnn();
  }
#endif
}


void ConvolutionComputation::
ConvolveForward(const CuMatrixBase<BaseFloat> &input,
                const CuMatrixBase<BaseFloat> &params,
                const CuVectorBase<BaseFloat> &bias,
                CuMatrixBase<BaseFloat> *output) const {
  const ConvolutionComputationConfig &c = config_;
  // Check some dimensions.
  KALDI_ASSERT(
      input.NumRows() == c.num_images * c.input_image_width &&
      input.NumCols() == c.input_image_height * c.num_channels_in &&
      input.Stride() == input.NumCols() &&
      params.NumRows() == c.num_channels_out &&
      params.NumCols() == c.num_channels_in * c.filter_height * c.filter_width &&
      params.Stride() == params.NumCols() &&
      bias.Dim() == c.num_channels_out &&
      output->NumRows() == c.num_images * c.input_image_height &&
      output->NumCols() == c.input_image_width * c.num_channels_out &&
      output->Stride() == output->NumCols());

#ifdef HAVE_CUDNN
  if (CuDevice::Instantiate().Enabled()) {
    CuVector<BaseFloat> temp_space(temp_space_required_forward_ /
                                   sizeof(BaseFloat), kUndefined);
    CUDNN_SAFE_CALL(cudnnConvolutionBiasActivationForward(
        CuDevice::Instantiate().GetCudnnHandle(),
        &ONE,
        input_desc_,
        input.Data(),
        params_desc_,
        params.Data(),
        conv_desc_,
        fwd_algo_,
        temp_space.Data(),
        temp_space.Dim() * sizeof(BaseFloat),
        &ZERO,
        output_desc_,
        output->Data(),
        bias_desc_,
        bias.Data(),
        activation_desc_,
        output_desc_,
        output->Data()));
  } else
#endif
  {
    ConvolveForward(input.Mat(), params.Mat(), bias.Vec(),
                    &(output->Mat()));
  }
}


void ConvolutionComputation::
ConvolveForward(const MatrixBase<BaseFloat> &input,
                const MatrixBase<BaseFloat> &params,
                const VectorBase<BaseFloat> &bias,
                MatrixBase<BaseFloat> *output) const {
  const ConvolutionComputationConfig &c = config_;
  // Check some dimensions.
  KALDI_ASSERT(
      input.NumRows() == c.num_images * c.input_image_width &&
      input.NumCols() == c.input_image_height * c.num_channels_in &&
      input.Stride() == input.NumCols() &&
      params.NumRows() == c.num_channels_out &&
      params.NumCols() == c.num_channels_in * c.filter_height * c.filter_width &&
      params.Stride() == params.NumCols() &&
      bias.Dim() == c.num_channels_out &&
      output->NumRows() == c.num_images * c.input_image_height &&
      output->NumCols() == c.input_image_width * c.num_channels_out &&
      output->Stride() == output->NumCols());


  {  // Deal with the bias.
    SubMatrix<BaseFloat> output_rearranged(
        output->Data(),
        c.num_images * c.input_image_width * c.input_image_height,
        c.num_channels_out, c.num_channels_out);
    output_rearranged.CopyRowsFromVec(bias);
  }

  Matrix<BaseFloat> params_rearranged(c.filter_width * c.filter_height,
                                      c.num_channels_out * c.num_channels_in,
                                      kUndefined, kStrideEqualNumCols);
  ConvertParams(params, &params_rearranged);

  // We're using variable names w (as in width) for horizontal positions and h
  // (as in height) for vertical positions.  This is perhaps not ideal.
  for (int32 output_w = 0; output_w < c.output_image_width; output_w++) {
    for (int32 output_h = 0; output_h < c.output_image_height; output_h++) {
      for (int32 filter_h = 0; filter_h < c.filter_height; filter_h++) {
        int32 filter_h_flipped = c.filter_height - 1 - filter_h;
        int32 input_h = output_h * c.filter_stride_vertical
            - c.zero_padding_vertical
            + filter_h * c.filter_dilation_vertical;
        if (input_h < 0 || input_h >= c.input_image_height)
          continue;
        for (int32 filter_w = 0; filter_w < c.filter_width; filter_w++) {
          int32 filter_w_flipped = c.filter_width - 1 - filter_w;
          int32 input_w = output_w * c.filter_stride_horizontal
              - c.zero_padding_horizontal
              + filter_w * c.filter_dilation_horizontal;

          if (input_w < 0 || input_w >= c.input_image_width)
            continue;

          const BaseFloat *params_data = params_rearranged.RowData(
              filter_w_flipped * c.filter_height + filter_h_flipped);
          SubMatrix<BaseFloat> this_params(params_data,
                                           c.num_channels_out,
                                           c.num_channels_in, c.num_channels_in);
          const BaseFloat *input_data = input.Data() +
              input_w * c.input_image_height * c.num_channels_in +
              input_h * c.num_channels_in;
          SubMatrix<BaseFloat> this_input_pixel(input_data,
                                                c.num_images,
                                                c.num_channels_in,
                                                c.num_channels_in);
          SubMatrix<BaseFloat> this_output_pixel(input_data,
                                                 c.num_images,
                                                 c.num_channels_in,
                                                 c.num_channels_in);
          this_output_pixel.AddMatMat(1.0, this_input_pixel, kNoTrans,
                                      this_params, kTrans, 1.0);
        }
      }
    }
  }
}

void ConvolutionComputation::
ConvolveBackwardData(const CuMatrixBase<BaseFloat> &params,
                     const CuMatrixBase<BaseFloat> &output_deriv,
                     CuMatrixBase<BaseFloat> *input_deriv) const {
#ifdef HAVE_CUDNN
  if (CuDevice::Instantiate().Enabled()) {
    CuVector<BaseFloat> temp_space(temp_space_required_backward_data_ /
                                   sizeof(BaseFloat), kUndefined);
    CUDNN_SAFE_CALL(cudnnConvolutionBackwardData(
        CuDevice::Instantiate().GetCudnnHandle(),
        &ONE,
        params_desc_,
        params.Data(),
        output_desc_,
        output_deriv.Data(),
        conv_desc_,
        bwd_data_algo_,
        temp_space.Data(),
        temp_space.Dim() * sizeof(BaseFloat),
        &ZERO,
        input_desc_,
        input_deriv->Data()));
  } else
#endif
  {
    // TODO
  }
}

void ConvolutionComputation::
ConvolveBackwardData(const MatrixBase<BaseFloat> &params,
                     const MatrixBase<BaseFloat> &output_deriv,
                     MatrixBase<BaseFloat> *input_deriv) const {
  // TODO
}



void ConvolutionComputation::
ConvolveBackwardParams(const CuMatrixBase<BaseFloat> &output_deriv,
                       const CuMatrixBase<BaseFloat> &input,
                       BaseFloat alpha,
                       CuMatrixBase<BaseFloat> *params_deriv) const {
#ifdef HAVE_CUDNN
  if (CuDevice::Instantiate().Enabled()) {
    CuVector<BaseFloat> temp_space(temp_space_required_backward_params_ /
                                   sizeof(BaseFloat), kUndefined);
    CUDNN_SAFE_CALL(cudnnConvolutionBackwardFilter(
        CuDevice::Instantiate().GetCudnnHandle(),
        &alpha,
        input_desc_,
        input.Data(),
        output_desc_,
        output_deriv.Data(),
        conv_desc_,
        bwd_filter_algo_,
        temp_space.Data(),
        temp_space.Dim() * sizeof(BaseFloat),
        &ONE,
        params_desc_,
        params_deriv->Data()));
  } else
#endif
  {
    ConvolveBackwardParams(output_deriv.Mat(), input.Mat(),
                           alpha, &(params_deriv->Mat()));
  }
}


void ConvolutionComputation::
ConvolveBackwardParams(const MatrixBase<BaseFloat> &output_deriv,
                       const MatrixBase<BaseFloat> &input,
                       BaseFloat alpha,
                       MatrixBase<BaseFloat> *params_deriv) const {
  // TODO
}


void ConvolutionComputation::
ConvolveBackwardBias(const CuMatrixBase<BaseFloat> &output_deriv,
                     BaseFloat alpha,
                     CuVectorBase<BaseFloat> *bias_deriv) const {
#ifdef HAVE_CUDNN
  if (CuDevice::Instantiate().Enabled()) {
    CUDNN_SAFE_CALL(cudnnConvolutionBackwardBias(
        CuDevice::Instantiate().GetCudnnHandle(),
        &alpha,
        output_desc_,
        output_deriv.Data(),
        &ONE,
        bias_desc_,
        bias_deriv->Data()));
  } else
#endif
  {
    ConvolveBackwardBias(output_deriv.Mat(), alpha, &(bias_deriv->Vec()));
  }
}

void ConvolutionComputation::
ConvolveBackwardBias(const MatrixBase<BaseFloat> &output_deriv,
                     BaseFloat alpha,
                     VectorBase<BaseFloat> *bias_deriv) const {
  // TODO.
}


// This function, called only if we are not using the GPU, converts
// the params from KWHC format to WHKC format (which is more convenient
// when using the CPU.  Note: K == channels-out, C == channels-in.
void ConvolutionComputation::ConvertParams(
    const MatrixBase<BaseFloat> &params,
    MatrixBase<BaseFloat> *params_rearranged) const {
  const ConvolutionComputationConfig &c = config_;
  KALDI_ASSERT(params.NumRows() == c.num_channels_out &&
               params.Stride() == params.NumCols() &&
               params_rearranged->NumRows() == c.filter_width * c.filter_height &&
               params_rearranged->Stride() == params_rearranged->NumCols());

  int32 num_rows_reinterpret =
      c.num_channels_out * c.filter_width * c.filter_height,
      num_cols_reinterpret = c.num_channels_in,
      area = c.filter_width * c.filter_height;

  // Reinterpret params as params_reinterpret which is of dimension KWH * C
  SubMatrix<BaseFloat> params_reinterpret(
      params.Data(),
      num_rows_reinterpret, num_cols_reinterpret, num_cols_reinterpret);
  SubMatrix<BaseFloat> params_rearranged_reinterpret(
      params_rearranged->Data(),
      num_rows_reinterpret, num_cols_reinterpret, num_cols_reinterpret);
  for (int32 k = 0; k < c.num_channels_out; k++) {
    for (int32 wh = 0; wh < area; wh++) {
      int32 params_row = k * area + wh,
          params_rearranged_row = wh * c.num_channels_out + k;
      SubVector<BaseFloat> src(params_reinterpret, params_row),
          dest(*params_rearranged, params_rearranged_row);
      dest.CopyFromVec(src);
    }
  }
}


}  // namespace cudnn_convolution
}  // namespace nnet3
}  // namespace kaldi
