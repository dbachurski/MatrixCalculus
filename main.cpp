#include <iostream>
#include <chrono>
#include "strassen.hpp"

using namespace Eigen;
using namespace std::chrono;

int main() {
    MatrixXd A = MatrixXd::Random(256, 256);
    MatrixXd B = MatrixXd::Random(256, 256);
    MatrixXd C_standard;
    MatrixXd C_strassen;
    unsigned int threshold;

    Strassen strassen;

    // Measure time for standard multiplication
    auto start_standard = high_resolution_clock::now();
    C_standard = A * B;
    auto end_standard = high_resolution_clock::now();
    auto duration_standard = duration_cast<microseconds>(end_standard - start_standard);
    std::cout << "Standard multiplication time: " << duration_standard.count() << " microseconds\n";

    // Measure time for Strassen multiplication (threshold = 2)
    threshold = 2;
    auto start_strassen = high_resolution_clock::now();
    C_strassen = strassen.strassenMultiply(A, B, threshold);
    auto end_strassen = high_resolution_clock::now();
    auto duration_strassen = duration_cast<microseconds>(end_strassen - start_strassen);
    std::cout << "Strassen multiplication time (threshold = 2): " << duration_strassen.count() << " microseconds\n";

    // Measure time for Strassen multiplication (threshold = 64)
    threshold = 64;
    start_strassen = high_resolution_clock::now();
    C_strassen = strassen.strassenMultiply(A, B, threshold);
    end_strassen = high_resolution_clock::now();
    duration_strassen = duration_cast<microseconds>(end_strassen - start_strassen);
    std::cout << "Strassen multiplication time (threshold = 64): " << duration_strassen.count() << " microseconds\n";

    // Calculate the difference between Strassen and standard multiplication results
    std::cout << "Difference norm: " << (C_strassen - C_standard).norm() << "\n";

    return 0;
}