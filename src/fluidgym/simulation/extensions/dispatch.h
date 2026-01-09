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

#ifndef DISPATCH_H
#define DISPATCH_H
#include <iostream>
#include "custom_types.h"
#include "transformations.h"
#include "grid_definitions.h"
#include "domain_structs.h"

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#define HOST __host__
#else
#define HOST
#define HOST_DEVICE
#endif

#ifdef __CUDACC__
static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
	if (err == cudaSuccess) return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
            << err << ") at " << file << ":" << line << std::endl;
	exit(10);
}

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
#endif

#define SWITCH_DIMS_CASE(DIM, ...) \
	case DIM: { \
		const index_t dim = DIM; \
		__VA_ARGS__; \
		break; \
	}

#define SWITCH_DIMS_SWITCH(DIMS, ...) \
	switch(DIMS) { \
		__VA_ARGS__ \
		default: \
			TORCH_CHECK(false, "Unknown dimension."); \
	}

#define SWITCH_DIMS(DIM, ...) \
	SWITCH_DIMS_SWITCH(DIM, SWITCH_DIMS_CASE(1, __VA_ARGS__) SWITCH_DIMS_CASE(2, __VA_ARGS__) SWITCH_DIMS_CASE(3, __VA_ARGS__))

// WARNING: do not declare int dim when using this macro to avoid aliasing problems
#define DISPATCH_FTYPES_DIMS(FTYPE, DIMS, NAME, ...) \
	AT_DISPATCH_FLOATING_TYPES(FTYPE, NAME, ([&] { \
		SWITCH_DIMS(DIMS, \
			__VA_ARGS__; \
		); \
	}));

inline index_t calc_num_spatial_points(const torch::Tensor &t){
	TORCH_CHECK(t.sizes().size() > 2, "tensor must have at least three dimensions");
	index_t num_points = 1;
	for(int dim = 2; dim < t.sizes().size(); dim++){
		num_points *= t.size(dim);
	}
	return num_points;
}

template <index_t DIMS>
inline HOST_DEVICE I4 make_slowest_running(const I4 &indices, index_t slowest){
	if(DIMS==1){
		return indices;
	}
	else if(DIMS==2){
		if(slowest==0){
			return {.a={indices.a[1], indices.a[0], 0, 0}};
		}
		else{
			return indices;
		}
	}
	else if(DIMS==3){
		if(slowest==0){
			return {.a={indices.a[2], indices.a[0], indices.a[1], 0}};
		}
		else if(slowest==1){
			return {.a={indices.a[0], indices.a[2], indices.a[1], 0}};
		}
		else{
			return indices;
		}
	}
}

#undef HOST
#undef HOST_DEVICE

#endif //ifndef DISPATCH_H
