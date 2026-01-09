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

#ifndef _INCLUDE_ORTHO_BASIS
#define _INCLUDE_ORTHO_BASIS

#include "custom_types.h"
//#include "optional.h"
#include <torch/extension.h>

torch::Tensor MakeBasisUnique(const torch::Tensor &basisMatrices, const torch::Tensor &sortingVectors, const bool inPlace);

#endif //_INCLUDE_ORTHO_BASIS