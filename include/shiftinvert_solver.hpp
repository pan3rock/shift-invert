#ifndef SHIFTINVERT_SOLVER_H_
#define SHIFTINVERT_SOLVER_H_

#include <Eigen/Dense>
#include <complex>

class ShiftinvertSolver {
public:
  ShiftinvertSolver(const Eigen::Ref<const Eigen::MatrixXd> &matA,
                    const Eigen::Ref<const Eigen::MatrixXd> &matB);
  std::complex<double> compute(double sigma, int niter);

private:
  const int ndim_;
  Eigen::MatrixXd matA_;
  Eigen::MatrixXd matB_;
};
#endif