//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <memory>
#include <vector>

#include "include/utils.h"

namespace PaddleDetection {

void SOLOv2PostProcess(
    const std::vector<cv::Mat> mats,
    std::vector<PaddleDetection::SOLOv2Result> *result,
    std::vector<int> out_bbox_num_data_,
    std::vector<int64_t> output_label_data_,
    std::vector<float> output_score_data_,
    std::vector<uint8_t> output_global_mask_data_);

} // namespace PaddleDetection
