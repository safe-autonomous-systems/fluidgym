/*
	Copyright 2025 Aleksandra Franz, Nils Thürey

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
*/

#pragma once

#ifndef _INCLUDE_RESAMPLING
#define _INCLUDE_RESAMPLING

#include "custom_types.h"
#include <torch/extension.h>

enum class BoundarySampling : int8_t{
	CONSTANT=0,
	CLAMP=1
	
};

torch::Tensor SampleTransformedGridGlobalToLocal(const torch::Tensor &globalData, const torch::Tensor &globalTransform, const torch::Tensor &localCoords,const BoundarySampling boundaryMode, const torch::Tensor &constantValue);
std::vector<torch::Tensor> SampleTransformedGridLocalToGlobal(const torch::Tensor &localData, const torch::Tensor &localCoords, const torch::Tensor &globalTransform, const torch::Tensor &globalShape, const index_t fillMaxSteps);
std::vector<torch::Tensor> SampleTransformedGridLocalToGlobalMulti(const std::vector<torch::Tensor> &localData, const std::vector<torch::Tensor> &localCoords, const torch::Tensor &globalTransform, const torch::Tensor &globalShape, const index_t fillMaxSteps);
torch::Tensor WorldPosFromGridPos(const torch::Tensor& vertex_grid, const torch::Tensor &grid_pos);
#endif //_INCLUDE_RESAMPLING