#include "ezsolver.hpp"

#include <Eigen/Dense>
#include <ezarpack/arpack_worker.hpp>
#include <ezarpack/storages/eigen.hpp>

using namespace Eigen;
using namespace ezarpack;

EzSolver::EzSolver(const Ref<const MatrixXd> &matA,
                   const Ref<const MatrixXd> &matB)
    : ndim_(matA.rows()), matA_(matA), matB_(matB) {}

VectorXcd EzSolver::compute(double sigma, int nev) {
  MatrixXd matL = matA_ - sigma * matB_;
  // auto lu = matL.partialPivLu();
  auto lhh = matL.householderQr();

  using worker_t = arpack_worker<ezarpack::Asymmetric, eigen_storage>;
  using params_t = worker_t::params_t;
  using vector_view_t = worker_t::vector_view_t;
  using vector_const_view_t = worker_t::vector_const_view_t;

  auto o = [&](vector_const_view_t x, vector_view_t y) {
    y = matB_ * x;
    y = lhh.solve(y).eval();
  };
  worker_t worker(ndim_);
  params_t params(nev, params_t::LargestMagnitude, params_t::Ritz);
  worker(o, params);
  auto const &mu = worker.eigenvalues();
  VectorXcd lambda = mu.cwiseInverse().eval();
  lambda.array() += sigma;
  return lambda;
}

VectorXcd EzSolver::compute_sym(double sigma, int nev) {
  MatrixXd matL = matA_ - sigma * matB_;
  // auto lu = matL.partialPivLu();
  auto lhh = matL.householderQr();

  using worker_t = arpack_worker<ezarpack::Symmetric, eigen_storage>;
  using params_t = worker_t::params_t;
  using vector_view_t = worker_t::vector_view_t;
  using vector_const_view_t = worker_t::vector_const_view_t;

  auto o = [&](vector_const_view_t x, vector_view_t y) {
    y = matB_ * x;
    y = lhh.solve(y).eval();
  };
  worker_t worker(ndim_);
  params_t params(nev, params_t::LargestMagnitude, false);
  worker(o, params);
  auto const &mu = worker.eigenvalues();
  VectorXcd lambda = mu.cwiseInverse().eval();
  lambda.array() += sigma;
  return lambda;
}