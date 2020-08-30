#ifndef EZSOLVER_H_
#define EZSOLVER_H_

#include <Eigen/Dense>

class EzSolver {
public:
  EzSolver(const Eigen::Ref<const Eigen::MatrixXd> &matA,
           const Eigen::Ref<const Eigen::MatrixXd> &matB);
  Eigen::VectorXcd compute(double sigma, int nev);
  Eigen::VectorXcd compute_sym(double sigma, int nev);

private:
  int ndim_;
  Eigen::MatrixXd matA_, matB_;
};

#endif