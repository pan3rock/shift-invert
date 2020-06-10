#include "shiftinvert_solver.hpp"

#include <Eigen/Dense>
#include <complex>

using namespace Eigen;

ShiftinvertSolver::ShiftinvertSolver(const Ref<const MatrixXd> &matA,
                                     const Ref<const MatrixXd> &matB)
    : ndim_(matA.rows()), matA_(matA), matB_(matB) {}

std::complex<double> ShiftinvertSolver::compute(double sigma, int niter) {
  MatrixXd matL = matA_ - sigma * matB_;
  VectorXd x = VectorXd::Random(ndim_);
  std::complex<double> lamb;
  for (int i = 0; i < niter; ++i) {
    VectorXd u = x / x.norm();
    VectorXd vecR = matB_ * u;
    x = matL.ldlt().solve(vecR);
    lamb = u.dot(x);
  }
  lamb = 1.0 / lamb + sigma;
  return lamb;
}