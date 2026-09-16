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

#ifndef _INCLUDE_PISO_MULTIBLOCK_CUDA
#define _INCLUDE_PISO_MULTIBLOCK_CUDA

#include "domain_structs.h"
#include "bicgstab_solver.h"

const int8_t NON_ORTHO_DIRECT_MATRIX = 1;
const int8_t NON_ORTHO_DIRECT_RHS = 2;
const int8_t NON_ORTHO_DIAGONAL_MATRIX = 4;
const int8_t NON_ORTHO_DIAGONAL_RHS = 8;
const int8_t NON_ORTHO_CENTER_MATRIX = 16;


/** Build the matrix for the advection-diffusion system (prediction step)
  * inputs:
  * - domain.viscosity
  * - block.velocity
  * - block.transform (inverse and determinant)
  * - boundary.velocity
  * - boundary.transform
  * outputs (writes to):
  * - domain.C: full sparse matrix (value, row, index)
  * - domain.A: diagonal of C
  */
void SetupAdvectionMatrixEulerImplicit(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
	const int8_t nonOrthoFlags, const bool forPassiveScalar, const index_t passiveScalarChannel);

/** Build the RHS of the prediction step for the passive scalar domain.scalarRHS. Adds non-orthogonal components using domain.scalarResult.
  * inputs:
  * - domain.viscosity
  * - domain.scalarResult (only if RHS nonOrthoFlags are set)
  * - block.passiveScalar
  * - block.transform
  * - boundary.passiveScalar
  * - boundary.velocity
  * - boundary.transform
  * outputs (writes to):
  * - domain.scalarRHS
  */
void SetupAdvectionScalarEulerImplicitRHS(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags);

/** Build the RHS of the prediction step for the velocity domain.velocityRHS. Adds non-orthogonal components using domain.velocityResult.
  * inputs:
  * - domain.viscosity
  * - domain.velocityResult (only if RHS nonOrthoFlags are set)
  * - block.velocity
  * - block.velocitySource
  * - block.pressure (if applyPressureGradient)
  * - block.transform
  * - boundary.velocity
  * - boundary.transform
  * outputs (writes to):
  * - domain.velocityRHS
  */
void SetupAdvectionVelocityEulerImplicitRHS(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool applyPressureGradient);

/** Build all 3 above.*/
//void SetupAdvectionEulerImplicitCombined(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool applyPressureGradient);

/** Build the whole pressure correction system:
  * domain.P: sparse matrix for linear system
  * domain.pressureRHS: RHS as vector field, buffered for velocity correction
  * domain.pressureRHSdiv: divergence of pressureRHS, the RHS used in the linear system. includes non-orthogonal components added after the divergence.
  */
void SetupPressureCorrection(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm);

/** Build only the pressure matrix.
  * inputs:
  * - domain.A
  * - block.transform
  * outputs (writes to):
  * - domain.P: full sparse matrix (value, row, index)
  */
void SetupPressureMatrix(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool useFaceTransform);

/** Build domain.pressureRHS and domain.pressureRHSdiv 
  * inputs:
  * - domain.A
  * - domain.C
  * - domain.velocityResult
  * - domain.pressureResult (only if RHS nonOrthoFlags are set)
  * - block.velocity
  * - block.velocitySource
  * - block.transform
  * - boundary.velocity
  * - boundary.transform
  * outputs (writes to):
  * - domain.pressureRHS
  * - domain.pressureRHSdiv
  */
void SetupPressureRHS(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm);

/** Build only domain.pressureRHSdiv, still including the non-orthogonal components.
  * inputs:
  * - domain.A (only if RHS nonOrthoFlags are set)
  * - domain.pressureRHS
  * - domain.pressureResult (only if RHS nonOrthoFlags are set)
  * - block.transform
  * - boundary.velocity
  * - boundary.transform
  * outputs (writes to):
  * - domain.pressureRHSdiv
  */
void SetupPressureRHSdiv(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm);

/** Make the velocity divergence free using the pressure gradient: velocityResult = pressureRHS - grad(pressure).
  * inputs:
  * - domain.A
  * - domain.pressureRHS (= predicted velocity)
  * - block.pressure
  * - block.transform
  * outputs (writes to):
  * - domain.velocityResult
  */
void CorrectVelocity(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const index_t version, const bool timeStepNorm);

torch::Tensor ComputeVelocityDivergence(std::shared_ptr<Domain> domain);
torch::Tensor ComputePressureGradient(std::shared_ptr<Domain> domain, const bool useFVM, const index_t gradientInterpolation);

/** Build the Laplacian matrix (no 1/A weighting) into domain->Epot.
 *  For thin-wall FIXED boundaries (epotCw > 0), the Robin BC
 *  dphi/dn = Cw * nabla^2_tau(phi) is incorporated directly as tangential
 *  surface Laplacian corrections — no augmented system is needed.
 *  Requires domain->Epot to be allocated via domain->CreateEpotMatrix(). */
void SetupEpotMatrix(std::shared_ptr<Domain> domain, const int8_t nonOrthoFlags, const bool useFaceTransform);

/** Compute the electric-potential Poisson RHS, div(u×B), for the inductionless MHD solve.
 *  vectorField: flat u×B field [totalSize * spatialDims], same layout as domain.pressureRHS.
 *  At a solid wall the normal flux is dropped (insulating, j_n=0); at an open boundary it is
 *  the cell-centre (u×B)_n (∂φ/∂n=0, current exits and closes virtually outside). Walls and
 *  open bounds are both BoundaryType::FIXED and are told apart by their velocity BC, see
 *  isInsulatingWallBound. Either way this matches the reconstruction in
 *  ComputeCurrentDensityFaceBased so that div(j)=0 holds discretely at boundary cells.
 *  Returns a new tensor of shape [totalSize]. */
torch::Tensor ComputeEpotRHS(std::shared_ptr<Domain> domain, const torch::Tensor &vectorField);

/** Compute the gradient of an arbitrary flat scalar field with Neumann (zero-gradient) BCs.
 *  scalarField: flat tensor of shape [totalSize], same layout as domain.pressureResult.
 *  Returns a new tensor of shape [totalSize * spatialDims]. */
torch::Tensor ComputeFieldGradient(std::shared_ptr<Domain> domain, const torch::Tensor &scalarField);

/** Compute FVM (Green-Gauss) gradient of an arbitrary flat scalar field.
 *  Uses face-based interpolation consistent with the Laplacian stencil.
 *  scalarField: flat tensor of shape [totalSize].
 *  Returns a new tensor of shape [totalSize * spatialDims]. */
torch::Tensor ComputeFieldGradientFVM(std::shared_ptr<Domain> domain, const torch::Tensor &scalarField);

/** Compute Lorentz force using face-based current density reconstruction.
 *  Guarantees discrete div(j)=0 by using the same stencil as the Poisson solve.
 *  epotField: flat [totalSize], uCrossBField: flat [totalSize * spatialDims or totalSize*3 for 2D].
 *  Returns current density tensor of shape [totalSize * 3]. */
torch::Tensor ComputeCurrentDensityFaceBased(std::shared_ptr<Domain> domain,
		const torch::Tensor &epotField, const torch::Tensor &uCrossBField);

std::vector<std::vector<torch::Tensor>> ComputeSpatialVelocityGradients(std::shared_ptr<Domain> domain);

/** Compute dot(transform.T, vector), or dot(transform.T_inv, vector) if inverse.
  * Output to new tensor.
  */
torch::Tensor TransformVectors(const torch::Tensor &vectors, const torch::Tensor &transforms, const bool inverse);

/** Copy from domain.scalarResult to block.scalarData. */
void CopyScalarResultToBlocks(std::shared_ptr<Domain> domain);

/** Copy from block.scalarData to domain.scalarResult. */
void CopyScalarResultFromBlocks(std::shared_ptr<Domain> domain);

/** Copy from domain.pressureResult to block.pressure. */
void CopyPressureResultToBlocks(std::shared_ptr<Domain> domain);

/** Copy from domain.epotResult to block.epot. */
void CopyEpotResultToBlocks(std::shared_ptr<Domain> domain);

/** Copy from block.epot to domain.epotResult. */
void CopyEpotResultFromBlocks(std::shared_ptr<Domain> domain);

/** Copy from block.pressure to domain.pressureResult. */
void CopyPressureResultFromBlocks(std::shared_ptr<Domain> domain);

/** Copy from domain.velocityResult to block.velocity. */
void CopyVelocityResultToBlocks(std::shared_ptr<Domain> domain);

/** Copy from block.velocity to domain.velocityResult. */
void CopyVelocityResultFromBlocks(std::shared_ptr<Domain> domain);

/** Compute the additive Smagorinsky SGS viscosities.
 *  NOTE: 'coefficient' is already C_s^2, and the filter width is the max cell edge length
 *  rather than cellVolume^(1/dims). SGSviscosityIncompressibleWALE does neither. */
std::vector<torch::Tensor> SGSviscosityIncompressibleSmagorinsky(std::shared_ptr<Domain> domain, const torch::Tensor coefficient);

/** Compute the additive WALE (Nicoud & Ducros 1999) SGS viscosities.
 *  'coefficient' is the textbook Cw (typ. 0.325-0.5) and is SQUARED internally; this differs from
 *  SGSviscosityIncompressibleSmagorinsky, whose coefficient is already C_s^2.
 *  Only meaningful for 3D domains: for dims<3 the WALE operator vanishes identically. */
std::vector<torch::Tensor> SGSviscosityIncompressibleWALE(std::shared_ptr<Domain> domain, const torch::Tensor coefficient);

solverReturn_t SolveLinear(std::shared_ptr<CSRmatrix> A, torch::Tensor RHS, torch::Tensor x, torch::Tensor maxit, torch::Tensor tol, const ConvergenceCriterion conv,
	const bool useBiCG, const bool matrixRankDeficient, const index_t residualResetSteps, const bool transposeA, const bool printResidual, const bool returnBestResult,
	const bool withPreconditioner);

/** Compute the outer product of vectors a and b. Write to result to out_pattern.value using its sparsity pattern. */
void SparseOuterProduct(torch::Tensor &a, torch::Tensor &b, std::shared_ptr<CSRmatrix> out_pattern);


/* --- Backwards/Gradient kernels --- 
 * Behaves like an "inverse" of the forward operations/kernels, output_grad->input_grad.
 * Not all gradient paths are implemented, meaning that not all forward inputs are differentiable (e.g. transformations metrics, boundary values).
 */
 
#ifdef WITH_GRAD
/**
  * inputs:
  * - domain.C_grad
  * - domain.A_grad
  * - block.transform
  * - boundary.transform
  * outputs:
  * - domain.viscosity_grad
  * - block.velocity_grad
  * - boundary.velocity_grad
  * not differentiable (outputs not implemented):
  * - block.transform_grad
  * - boundary.transform_grad
  */
void SetupAdvectionMatrixEulerImplicit_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
	const int8_t nonOrthoFlags, const bool forPassiveScalar, const index_t passiveScalarChannel);

/**
  * inputs:
  * - domain.scalarRHS_grad
  * - domain.viscosity
  * - block.transform
  * - boundary.passiveScalar
  * - boundary.velocity
  * - boundary.transform
  * outputs:
  * - domain.viscosity_grad
  * - domain.scalarResult_grad (only if RHS nonOrthoFlags are set)
  * - block.passiveScalar_grad
  * - boundary.passiveScalar_grad
  * - boundary.velocity_grad
  * not differentiable:
  * - block.transform_grad
  * - boundary.transform_grad
  */
void SetupAdvectionScalarEulerImplicitRHS_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags);

/**
  * inputs:
  * - domain.velocityRHS_grad
  * - domain.viscosity
  * - block.transform
  * - boundary.velocity
  * - boundary.transform
  * outputs:
  * - domain.viscosity_grad
  * - domain.velocityResult_grad (only if RHS nonOrthoFlags are set)
  * - block.velocity_grad
  * - block.velocitySource_grad
  * - boundary.velocity_grad
  * not differentiable:
  * - block.pressure_grad (TODO?, only if applyPressureGradient)
  * - block.transform_grad
  * - boundary.transform_grad
  */
void SetupAdvectionVelocityEulerImplicitRHS_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool applyPressureGradient);

/** Backwards pass for: non-orthogonal RHS components (TODO), pressureRHSdiv, pressureRHS, pressure matrix (not implemented)
  * inputs/outputs: identical to SetupPressureRHS_GRAD
  * not differentiable:
  * - (complete pressure matrix setup, only uses domain.A and transforms)
  */
void SetupPressureCorrection_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
	const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm);

void SetupPressureMatrix_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
	const int8_t nonOrthoFlags, const bool useFaceTransform);

/** Backwards pass for: non-orthogonal RHS components (TODO), pressureRHSdiv, pressureRHS
  * inputs:
  * - domain.C
  * - domain.A
  * - domain.viscosity
  * - domain.pressureRHS_grad
  * - domain.pressureRHSdiv_grad
  * - block.transform
  * - boundary.velocity
  * - boundary.transform
  * outputs:
  * - domain.pressureResult_grad (only if RHS nonOrthoFlags are set)
  * - domain.pressureRHS_grad
  * - domain.velocityResult_grad
  * - domain.viscosity_grad
  * - block.velocity_grad
  * - block.velocitySource_grad
  * not differentiable:
  * - domain.C_grad (TODO)
  * - domain.A_grad (TODO)
  */
void SetupPressureRHS_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
	const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm);

/** Backwards pass for: non-orthogonal RHS components (TODO), pressureRHSdiv
  * inputs:
  * - domain.pressureRHSdiv_grad
  * - block.transform
  * outputs:
  * - domain.pressureResult_grad (TODO, only if RHS nonOrthoFlags are set)
  * - domain.pressureRHS_grad
  * not differentiable:
  * - block.transform_grad
  */
void SetupPressureRHSdiv_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
	const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm);

/**
  * inputs:
  * - domain.A
  * - domain.velocityResult_grad
  * - block.transform
  * outputs:
  * - domain.pressureRHS_grad
  * - block.pressure_grad
  * not differentiable:
  * - domain.A_grad
  */
void CorrectVelocity_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const bool timeStepNorm);

/** Copy from block.passiveScalar_grad to domain.scalarResult_grad. */
void CopyScalarResultGradFromBlocks(std::shared_ptr<Domain> domain);

/** Copy from domain.scalarResult_grad to block.passiveScalar_grad. */
void CopyScalarResultGradToBlocks(std::shared_ptr<Domain> domain);
void CopyPressureResultGradFromBlocks(std::shared_ptr<Domain> domain);
void CopyVelocityResultGradFromBlocks(std::shared_ptr<Domain> domain);
void CopyVelocityResultGradToBlocks(std::shared_ptr<Domain> domain);
void CopyEpotResultGradFromBlocks(std::shared_ptr<Domain> domain);

torch::Tensor ComputeEpotRHSGrad(std::shared_ptr<Domain> domain,
		const torch::Tensor &gradDivergence);

std::pair<torch::Tensor, torch::Tensor> ComputeCurrentDensityFaceBasedGrad(
		std::shared_ptr<Domain> domain,
		const torch::Tensor &epotField,
		const torch::Tensor &uCrossBField,
		const torch::Tensor &gradJ);
#endif //WITH_GRAD

#endif //_INCLUDE_PISO_MULTIBLOCK_CUDA