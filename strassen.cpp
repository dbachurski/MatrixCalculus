#include "strassen.hpp"

MatrixXd Strassen::strassenMultiply(const MatrixXd& A, const MatrixXd& B, unsigned int threshold = 64) {
    assert(A.cols() == B.rows() && "Matrix dimensions must match");
    assert((threshold & (threshold - 1)) == 0 && "Threshold must be a power of 2");

    int originalRows = A.rows();
    int originalCols = B.cols();
    int size = std::max(A.rows(), std::max(A.cols(), B.cols()));
    int paddedSize = nextPowerOfTwo(size);

    // Fill the matrices with zeros up to dimensions equal power of 2
    MatrixXd A_pad = MatrixXd::Zero(paddedSize, paddedSize);
    A_pad.block(0, 0, A.rows(), A.cols()) = A;
    MatrixXd B_pad = MatrixXd::Zero(paddedSize, paddedSize);
    B_pad.block(0, 0, B.rows(), B.cols()) = B;

    // Multiply matrices using Strassen method
    MatrixXd C_pad = strassen(A_pad, B_pad, threshold);
    MatrixXd C = C_pad.block(0, 0, originalRows, originalCols);

    return C;
}

unsigned int Strassen::nextPowerOfTwo(unsigned int n) {
    unsigned int power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

MatrixXd Strassen::strassen(const MatrixXd& A, const MatrixXd& B, unsigned int threshold) {
    int n = A.rows();
    int half = n / 2;

    // Base case: use standard multiplication for matrices smaller than or equal to the threshold
    if (n <= threshold) {
        return A * B;
    }

    auto A11 = A.block(0, 0, half, half);
    auto A12 = A.block(0, half, half, half);
    auto A21 = A.block(half, 0, half, half);
    auto A22 = A.block(half, half, half, half);

    auto B11 = B.block(0, 0, half, half);
    auto B12 = B.block(0, half, half, half);
    auto B21 = B.block(half, 0, half, half);
    auto B22 = B.block(half, half, half, half);

    MatrixXd M1 = strassen(A11 + A22, B11 + B22, threshold);
    MatrixXd M2 = strassen(A21 + A22, B11, threshold);
    MatrixXd M3 = strassen(A11, B12 - B22, threshold);
    MatrixXd M4 = strassen(A22, B21 - B11, threshold);
    MatrixXd M5 = strassen(A11 + A12, B22, threshold);
    MatrixXd M6 = strassen(A21 - A11, B11 + B12, threshold);
    MatrixXd M7 = strassen(A12 - A22, B21 + B22, threshold);

    MatrixXd C11 = M1 + M4 - M5 + M7;
    MatrixXd C12 = M3 + M5;
    MatrixXd C21 = M2 + M4;
    MatrixXd C22 = M1 - M2 + M3 + M6;

    MatrixXd C(n, n);
    C.block(0, 0, half, half) = C11;
    C.block(0, half, half, half) = C12;
    C.block(half, 0, half, half) = C21;
    C.block(half, half, half, half) = C22;

    return C;
}
