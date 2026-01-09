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

#ifndef _INCLUDE_MATRIX_VECTOR_OPS
#define _INCLUDE_MATRIX_VECTOR_OPS

#include "custom_types.h"
//#include "optional.h"
#include <torch/extension.h>

torch::Tensor matmul(
	const torch::Tensor &vectorMatrixA, const torch::Tensor &vectorMatrixB,
	const bool transposeA, const bool invertA,
	const bool transposeB, const bool invertB,
	const bool transposeOutput, const bool invertOutput
);
std::vector<torch::Tensor> matmulGrad(
	const torch::Tensor &vectorMatrixA, const torch::Tensor &vectorMatrixB, const torch::Tensor &outputGrad,
	const bool transposeA, const bool invertA,
	const bool transposeB, const bool invertB,
	const bool transposeOutput, const bool invertOutput
);

torch::Tensor VectorToDiagMatrix(const torch::Tensor &vectors);
torch::Tensor InvertMatrix(const torch::Tensor &matrices, const bool inPlace);

// CUDA device code, only enable when compiling cuda code
#ifdef __CUDACC__

#include "transformations.h"

template<typename scalar_t, int DIMS>
__device__ inline
Vector<scalar_t, DIMS> loadVectorFromChannels(const scalar_t* vectors, const index_t flatPos, const index_t stride){
	Vector<scalar_t, DIMS> v;
	for(index_t i=0; i<DIMS; ++i){
		v.a[i] = vectors[flatPos + i*stride];
	}
	return v;
}

template<typename scalar_t, int DIMS>
__device__ inline
void writeVectorToChannels(const Vector<scalar_t, DIMS> &v, scalar_t* vectors, const index_t flatPos, const index_t stride){
	for(index_t i=0; i<DIMS; ++i){
		vectors[flatPos + i*stride] = v.a[i];
	}
}

template<typename scalar_t, int DIMS>
__device__ inline
MatrixSquare<scalar_t, DIMS> loadMatrixFromChannels(const scalar_t* matrices, const index_t flatPos, const index_t stride){
	MatrixSquare<scalar_t, DIMS> m;
	for(index_t row=0; row<DIMS; ++row){
		for(index_t col=0; col<DIMS; ++col){
			m.a[row][col] = matrices[flatPos + (row*DIMS + col)*stride];
		}
	}
	return m;
}

template<typename scalar_t, int DIMS>
__device__ inline
void writeMatrixToChannels(const MatrixSquare<scalar_t, DIMS> &m, scalar_t* matrices, const index_t flatPos, const index_t stride){
	for(index_t row=0; row<DIMS; ++row){
		for(index_t col=0; col<DIMS; ++col){
			matrices[flatPos + (row*DIMS + col)*stride] = m.a[row][col];
		}
	}
}

template<typename scalar_t, int DIMS>
__device__ inline
MatrixSquare<scalar_t, DIMS> loadMatrixFromChannelsTransposed(const scalar_t* matrices, const index_t flatPos, const index_t stride){
	MatrixSquare<scalar_t, DIMS> m;
	for(index_t row=0; row<DIMS; ++row){
		for(index_t col=0; col<DIMS; ++col){
			m.a[col][row] = matrices[flatPos + (row*DIMS + col)*stride];
		}
	}
	return m;
}

template<typename scalar_t, int DIMS>
__device__ inline
void writeMatrixToChannelsTransposed(const MatrixSquare<scalar_t, DIMS> &m, scalar_t* matrices, const index_t flatPos, const index_t stride){
	for(index_t row=0; row<DIMS; ++row){
		for(index_t col=0; col<DIMS; ++col){
			matrices[flatPos + (row*DIMS + col)*stride] = m.a[col][row];
		}
	}
}

#endif //__CUDACC__

#endif //_INCLUDE_MATRIX_VECTOR_OPS