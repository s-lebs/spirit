#pragma once
#ifndef SPIRIT_CORE_ENGINE_EIGENMODES_HPP
#define SPIRIT_CORE_ENGINE_EIGENMODES_HPP

#include "Spirit_Defines.h"
#include <data/Geometry.hpp>
#include <data/Parameters_Method.hpp>
#include <data/Spin_System.hpp>
#include <utility/Logging.hpp>

#include <memory>
#include <vector>

namespace Engine
{
namespace Eigenmodes
{

// Check whether system members and EMA parameters are consistent with eachother
void Check_Eigenmode_Parameters( std::shared_ptr<Data::Spin_System> system );

// Calculate a systems eigenmodes according to its EMA parameters
void Calculate_Eigenmodes( std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain );

// Calculate the full eigenspectrum of a Hessian (needs to be self-adjoint)
// gradient and hessian should be the 3N-dimensional representations without constraints
bool Hessian_Full_Spectrum(
    const std::shared_ptr<Data::Parameters_Method> parameters, const vectorfield & spins, const vectorfield & gradient,
    const MatrixX & hessian, MatrixX & tangent_basis, MatrixX & hessian_constrained, VectorX & eigenvalues,
    MatrixX & eigenvectors );

// Calculate a partial eigenspectrum of a Hessian
// gradient and hessian should be the 3N-dimensional representations without constraints
bool Hessian_Partial_Spectrum(
    const std::shared_ptr<Data::Parameters_Method> parameters, const vectorfield & spins, const vectorfield & gradient,
    const MatrixX & hessian, std::size_t n_modes, MatrixX & tangent_basis, MatrixX & hessian_constrained, VectorX & eigenvalues,
    MatrixX & eigenvcetors );

// Calculate a partial eigenspectrum of a sparse Hessian
// gradient and hessian should be the 3N-dimensional representations without constraints
bool Sparse_Hessian_Partial_Spectrum(
    const std::shared_ptr<Data::Parameters_Method> parameters, const vectorfield & spins, const vectorfield & gradient,
    const SpMatrixX & hessian, int n_modes, SpMatrixX & tangent_basis, SpMatrixX & hessian_constrained, VectorX & eigenvalues,
    MatrixX & eigenvectors
);
// Calculate a partial eigenspectrum of the sparse Hessian with Gradient Decent
bool computeLowEV(const std::shared_ptr<Data::Parameters_Method> parameters, const vectorfield & spins, const vectorfield & gradient,
    const SpMatrixX & hessian, int n_modes, SpMatrixX & tangent_basis, SpMatrixX & hessian_constrained,
    VectorX & eigenvalues, MatrixX & eigenvectors, MatrixX & prev, int GDIterations);

//Transfer the 2N eigenvectors into the 3N eigenvectors
void Transfer_Eigenmodes( std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain );
} // end namespace Eigenmodes
} // end namespace Engine

#endif
