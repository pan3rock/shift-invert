#include "catch.hpp"
#include "shiftinvert_solver.hpp"
#include "timer.hpp"

#include <Eigen/Dense>
#include <complex>
#include <fmt/format.h>
#include <iostream>

using namespace Eigen;
using namespace std::complex_literals;

TEST_CASE("run through", "[solver]") {
  int ndim = 4;
  MatrixXd matA = MatrixXd::Random(ndim, ndim);
  MatrixXd matB = MatrixXd::Random(ndim, ndim);
  GeneralizedEigenSolver<MatrixXd> ges(matA, matB);
  VectorXcd ev = ges.eigenvalues();
  for (int i = 0; i < ev.size(); ++i) {
    fmt::print("({:12.5f}, {:12.5f})\n", ev(i).real(), ev(i).imag());
  }
  int maxiter = 20;
  double tol = 1.0e-5;
  ShiftinvertSolver sis(matA, matB, maxiter, tol);
  std::complex<double> sigma = 0.59;
  std::complex<double> ev2 = sis.compute(sigma);
  fmt::print("eigenvalues near to ({:12.5f}, {:12.5f}): ({:12.5f}, {:12.5f})\n",
             sigma.real(), sigma.imag(), ev2.real(), ev2.imag());
  sigma = 2.8;
  ev2 = sis.compute(sigma);
  fmt::print("eigenvalues near to ({:12.5f}, {:12.5f}): ({:12.5f}, {:12.5f})\n",
             sigma.real(), sigma.imag(), ev2.real(), ev2.imag());
  sigma = 0.3 + 0.5i;
  ev2 = sis.compute(sigma);
  fmt::print("eigenvalues near to ({:12.5f}, {:12.5f}): ({:12.5f}, {:12.5f})\n",
             sigma.real(), sigma.imag(), ev2.real(), ev2.imag());
}

TEST_CASE("performance", "[solver]") {
  int ndim = 20;
  MatrixXd matA = MatrixXd::Random(ndim, ndim);
  MatrixXd matB = MatrixXd::Random(ndim, ndim);
  Timer::begin("GeneralizedEigenSolver");
  GeneralizedEigenSolver<MatrixXd> ges(matA, matB);
  VectorXcd ev = ges.eigenvalues();
  Timer::end("GeneralizedEigenSolver");
  for (int i = 0; i < ev.size(); ++i) {
    fmt::print("({:12.5f}, {:12.5f})\n", ev(i).real(), ev(i).imag());
  }
  Timer::begin("shift-invert");
  int maxiter = 20;
  double tol = 1.0e-5;
  ShiftinvertSolver sis(matA, matB, maxiter, tol);
  std::complex<double> sigma = 0.59;
  std::complex<double> ev2 = sis.compute(sigma);
  Timer::end("shift-invert");
  fmt::print("eigenvalues near to ({:12.5f}, {:12.5f}): ({:12.5f}, {:12.5f})\n",
             sigma.real(), sigma.imag(), ev2.real(), ev2.imag());
  std::cout << Timer::summery() << std::endl;
}