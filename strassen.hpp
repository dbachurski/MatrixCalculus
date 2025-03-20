#ifndef STRASSEN_HPP
#define STRASSEN_HPP

#include <Eigen/Dense>
#include <cassert>
#include <cmath>

using namespace Eigen;

class Strassen {
public:
    MatrixXd strassenMultiply(const MatrixXd& A, const MatrixXd& B, unsigned int threshold);

private:
    unsigned int nextPowerOfTwo(unsigned int n);
    MatrixXd strassen(const MatrixXd& A, const MatrixXd& B, unsigned int threshold);
};

#endif