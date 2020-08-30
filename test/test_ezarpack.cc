#include "catch.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <ezarpack/arpack_worker.hpp>
#include <ezarpack/storages/eigen.hpp>
#include <iostream>

using namespace Eigen;
using namespace ezarpack;

TEST_CASE("ok", "[ezarpack]") {
  int ndim = 10;
  MatrixXd matA = MatrixXd::Random(ndim, ndim);
  MatrixXd matB = MatrixXd::Random(ndim, ndim);
  matA = matA * matA.transpose();
  matB = matB * matB.transpose();
  matB += ndim * MatrixXd::Identity(ndim, ndim);
  matA = -matA;
  matB = -matB;

  using worker_t = arpack_worker<ezarpack::Symmetric, eigen_storage>;
  using vector_view_t = worker_t::vector_view_t;
  using vector_const_view_t = worker_t::vector_const_view_t;

  auto op = [&](vector_view_t from, vector_view_t to) { to = matA * from; };
  auto b = [&](vector_const_view_t from, vector_view_t to) {
    to = matB * from;
  };

  worker_t worker(ndim);

  using params_t = worker_t::params_t;
  int nev = ndim - 1;
  params_t params(nev, params_t::Smallest, false);

  worker(b, params);

  std::cout << "Eigenvalues (Ritz values):" << std::endl;
  std::cout << worker.eigenvalues().transpose() << std::endl;

  EigenSolver<MatrixXd> es(matB, false);
  std::cout << es.eigenvalues().transpose().real() << std::endl;

  worker_t worker2(ndim);

  double sigma = 0.5;
  MatrixXd mat_op = matA - sigma * matB;

  auto op2 = [&](vector_view_t from, vector_view_t to) {
    to = matB * from;
    to = mat_op.ldlt().solve(to);
  };
  params.sigma = sigma;
  worker2(op2, b, worker_t::ShiftAndInvert, params);

  std::cout << "Eigenvalues (Ritz values):" << std::endl;
  std::cout << worker2.eigenvalues().transpose() << std::endl;

  GeneralizedEigenSolver<MatrixXd> es2(matA, matB, false);
  std::cout << es2.eigenvalues().transpose().real() << std::endl;
}