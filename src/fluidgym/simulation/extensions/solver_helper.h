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

#ifndef _INCLUDE_SOLVER_HELPER
#define _INCLUDE_SOLVER_HELPER

//#include <cuda.h>
//#include <cuda_runtime.h>
#include <cublas_v2.h>

// defined in cg_solver_kernel.cu
template< typename scalar_t>
extern scalar_t ComputeConvergenceCriterion(cublasHandle_t cublasHandle, const scalar_t *r, const index_t n, const ConvergenceCriterion conv);

#endif //_INCLUDE_SOLVER_HELPER