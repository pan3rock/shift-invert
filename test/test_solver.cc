#include "catch.hpp"
#include "ezsolver.hpp"
#include "shiftinvert_solver.hpp"
#include "timer.hpp"

#include <Eigen/Dense>
#include <complex>
#include <fmt/format.h>
#include <iostream>
#include <string>

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

void start_performance(int ndim, double sigma_in) {
  MatrixXd matA = MatrixXd::Random(ndim, ndim);
  MatrixXd matB = MatrixXd::Random(ndim, ndim);

  Timer::begin("GeneralizedEigenSolver");
  GeneralizedEigenSolver<MatrixXd> ges(matA, matB);
  VectorXcd ev = ges.eigenvalues();
  Timer::end("GeneralizedEigenSolver");
  fmt::print("\nGeneralizedEigenSolver\n");
  for (int i = 0; i < ev.size(); ++i) {
    fmt::print("({:12.5f}, {:12.5f})\n", ev(i).real(), ev(i).imag());
  }
  fmt::print("---------------------------------------------\n");

  Timer::begin("shift-invert");
  int maxiter = 100;
  double tol = 1.0e-5;
  ShiftinvertSolver sis(matA, matB, maxiter, tol);
  std::complex<double> sigma = sigma_in;
  std::complex<double> ev2 = sis.compute(sigma);
  Timer::end("shift-invert");
  fmt::print("eigenvalues near to ({:12.5f}, {:12.5f}):\n", sigma.real(),
             sigma.imag());
  fmt::print("({:12.5f}, {:12.5f})\n", ev2.real(), ev2.imag());
  fmt::print("---------------------------------------------\n");

  Timer::begin("ezsolver");
  EzSolver ezs(matA, matB);
  int nev = 4;
  double sigma3 = sigma_in;
  VectorXcd ev3 = ezs.compute(sigma3, nev);
  Timer::end("ezsolver");
  fmt::print("eigenvalues near to {:12.5f}:\n", sigma3);
  for (int i = 0; i < nev; ++i) {
    fmt::print("({:12.5f}, {:12.5f})\n", ev3(i).real(), ev3(i).imag());
  }
  fmt::print("---------------------------------------------\n");

  std::cout << Timer::summery() << std::endl;
}

TEST_CASE("performance 20", "[solver]") { start_performance(20, 0.59); }

TEST_CASE("performance 100", "[solver]") { start_performance(100, 0.59); }

TEST_CASE("performance 1000", "[solver]") { start_performance(1000, 0.59); }

TEST_CASE("ezsolver performance", "[solver]") {
  int ndim = 100;
  MatrixXd matA = MatrixXd::Random(ndim, ndim);
  MatrixXd matB = MatrixXd::Random(ndim, ndim);
  double sigma = 0.1;
  EzSolver ezs(matA, matB);
  VectorXi nevs(5);
  nevs << 1, 10, 20, 40, 80;
  for (int i = 0; i < nevs.size(); ++i) {
    int nev = nevs(i);
    std::string tag = fmt::format("nev = {:d}", nev);
    Timer::begin(tag);
    VectorXcd ev = ezs.compute(sigma, nev);
    Timer::end(tag);
    fmt::print("{:s}\n", tag);
    for (int i = 0; i < nev; ++i) {
      fmt::print("({:12.5f}, {:12.5f})\n", ev(i).real(), ev(i).imag());
    }
    fmt::print("----------------------------\n");
  }
  std::cout << Timer::summery() << std::endl;
}