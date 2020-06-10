#ifndef SHIFTINVERT_SOLVER_H_
#define SHIFTINVERT_SOLVER_H_

#include <Eigen/Dense>
#include <complex>

class ShiftinvertSolver {
public:
  ShiftinvertSolver(const Eigen::Ref<const Eigen::MatrixXd> &matA,
                    const Eigen::Ref<const Eigen::MatrixXd> &matB, int maxiter,
                    double tol);
  std::complex<double> compute(std::complex<double> sigma);

private:
  const int ndim_;
  Eigen::MatrixXd matA_;
  Eigen::MatrixXd matB_;
  const int maxiter_;
  const double tol_;
};
#endif