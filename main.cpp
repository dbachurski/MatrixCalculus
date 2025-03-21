#include <iostream>
#include <chrono>
#include <vector>
#include "strassen.hpp"

using namespace Eigen;
using namespace std::chrono;

struct OpCount {
    long multiplications;
    long additions;
};

OpCount countOperations(const std::string& algorithm, unsigned int n, unsigned int threshold = 0) {
    if (algorithm == "standard") {
        long mult = static_cast<long>(n) * n * n;
        long add = mult - static_cast<long>(n) * n;
        return {mult, add};
    } else if (algorithm == "strassen") {
        if (n <= threshold) {
            return countOperations("standard", n);
        } else {
            OpCount sub = countOperations("strassen", n / 2, threshold);
            long mult = 7 * sub.multiplications;
            long add = 7 * sub.additions + 18 * (n/2) * (n/2);
            return {mult, add};
        }
    } else {
        throw std::invalid_argument("Invalid algorithm name");
    }
}

void printTime(const std::string& label, long microseconds) {
    std::cout << label << ": " << microseconds << " microseconds\n";
}

void printOpCount(const std::string& label, const OpCount& opCount) {
    std::cout << label << " operations:\n";
    std::cout << "  Multiplications: " << opCount.multiplications << "\n";
    std::cout << "  Additions/Subtractions: " << opCount.additions << "\n";
}

int main() {
    // Grid: matrix sizes and thresholds
    std::vector<unsigned int> matrixSizes = {4, 8, 16, 32, 128, 256, 512};
    std::vector<unsigned int> thresholds = {4, 16, 64, 128};

    Strassen strassen;

    for (unsigned int n : matrixSizes) {
        std::cout << "\n=============================================\n";
        std::cout << "Matrix size: " << n << "x" << n << "\n";

        // Generate random matrix
        MatrixXd A = MatrixXd::Random(n, n);
        MatrixXd B = MatrixXd::Random(n, n);

        // Standard multiplication
        auto start_standard = high_resolution_clock::now();
        MatrixXd C_standard = A * B;
        auto end_standard = high_resolution_clock::now();
        auto duration_standard = duration_cast<microseconds>(end_standard - start_standard);
        printTime("Standard multiplication time", duration_standard.count());
        OpCount standard_ops = countOperations("standard", n);
        printOpCount("Standard multiplication", standard_ops);

        // Strassen multiplication for different thresholds
        for (unsigned int t : thresholds) {
            std::cout << "\n--- Strassen multiplication with threshold = " << t << " ---\n";
            auto start_strassen = high_resolution_clock::now();
            MatrixXd C_strassen = strassen.strassenMultiply(A, B, t);
            auto end_strassen = high_resolution_clock::now();
            auto duration_strassen = duration_cast<microseconds>(end_strassen - start_strassen);
            printTime("Strassen multiplication time", duration_strassen.count());
            OpCount strassen_ops = countOperations("strassen", n, t);
            printOpCount("Strassen multiplication", strassen_ops);
            std::cout << "Difference norm: " << (C_strassen - C_standard).norm() << "\n";
        }
    }

    return 0;
}
