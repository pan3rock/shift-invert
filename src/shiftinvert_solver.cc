#include "shiftinvert_solver.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <fmt/format.h>

using namespace Eigen;

ShiftinvertSolver::ShiftinvertSolver(const Ref<const MatrixXd> &matA,
                                     const Ref<const MatrixXd> &matB,
                                     int maxiter, double tol)
    : ndim_(matA.rows()), matA_(matA), matB_(matB), maxiter_(maxiter),
      tol_(tol) {}

std::complex<double> ShiftinvertSolver::compute(std::complex<double> sigma) {
  MatrixXcd matL = matA_ - sigma * matB_;
  auto lhh = matL.householderQr();
  VectorXcd x = VectorXcd::Random(ndim_);
  std::complex<double> lamb;
  std::complex<double> lamb_pre = 1.0e10;
  for (int i = 0; i < maxiter_; ++i) {
    VectorXcd u = x / x.norm();
    VectorXcd vecR = matB_ * u;
    x = lhh.solve(vecR);
    lamb = u.dot(x);
    lamb = 1.0 / lamb + sigma;
    double diff = std::abs(lamb - lamb_pre);
    fmt::print("{:d} {:12.5f}{:12.5f}({:12.5e})\n", i, lamb.real(), lamb.imag(),
               diff);
    if (diff < tol_) {
      break;
    } else {
      lamb_pre = lamb;
    }
  }
  return lamb;
}