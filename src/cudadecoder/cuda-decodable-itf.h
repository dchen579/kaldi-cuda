// cudadecoder/cuda-decodable-itf.h
/*
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 * Authors:  Hugo Braun, Justin Luitjens, Ryan Leary
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef KALDI_DECODABLE_ITF_H
#define KALDI_DECODABLE_ITF_H

#include "itf/decodable-itf.h"

namespace kaldi {

class CudaDecodableInterface : public DecodableInterface {
public:
  virtual BaseFloat *GetLogLikelihoodsCudaPointer(int32 subsampled_frame) = 0;
};
}

#endif