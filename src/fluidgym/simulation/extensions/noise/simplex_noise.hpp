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

#ifndef _INCLUDE_NOISE_SETTINGS
#define _INCLUDE_NOISE_SETTINGS
#include <torch/extension.h>
#include <vector>


namespace SimplexNoise{
enum NoiseVariation{
	SIMPLEX=0,
	WORLEY=1,
	FRACTAL_BROWNIAN_MOTION=2,
	RIDGED=3,
	RIDGED_MULTI_FRACTAL=4,
	GRADIENT=5,
	GRADIENT_FBM=6,
	CURL=7,
	CURL_FBM=8
};


torch::Tensor GenerateSimplexNoiseVariation(const std::vector<int32_t> output_shape, const torch::Device GPUdevice,
	const std::vector<float> scale, const std::vector<float> offset,
	const NoiseVariation variation, const float ridgeOffset, const int32_t octaves, const float lacunarity, const float gain);

} //namespace SimplexNoise

#endif //_INCLUDE_NOISE_SETTINGS