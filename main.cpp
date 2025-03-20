#include <iostream>
#include <chrono>
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
    MatrixXd A = MatrixXd::Random(256, 256);
    MatrixXd B = MatrixXd::Random(256, 256);
    MatrixXd C_standard, C_strassen;
    unsigned int threshold;

    Strassen strassen;

    // Multiply matrices using both standard and Strassen algorithms
    // Measure execution time for each method

    // Standard multiplication
    auto start_standard = high_resolution_clock::now();
    C_standard = A * B;
    auto end_standard = high_resolution_clock::now();
    auto duration_standard = duration_cast<microseconds>(end_standard - start_standard);
    printTime("Standard multiplication time", duration_standard.count());

    // Strassen multiplication (threshold = 2)
    threshold = 2;
    auto start_strassen = high_resolution_clock::now();
    C_strassen = strassen.strassenMultiply(A, B, threshold);
    auto end_strassen = high_resolution_clock::now();
    auto duration_strassen = duration_cast<microseconds>(end_strassen - start_strassen);
    printTime("Strassen multiplication time (threshold = 2)", duration_strassen.count());

    // Strassen multiplication (threshold = 64)
    threshold = 64;
    start_strassen = high_resolution_clock::now();
    C_strassen = strassen.strassenMultiply(A, B, threshold);
    end_strassen = high_resolution_clock::now();
    duration_strassen = duration_cast<microseconds>(end_strassen - start_strassen);
    printTime("Strassen multiplication time (threshold = 64)", duration_strassen.count());

    // Calculate number of operations
    OpCount standard_ops = countOperations("standard", 256);
    printOpCount("\nStandard multiplication", standard_ops);

    OpCount strassen_ops_2 = countOperations("strassen", 256, 2);
    printOpCount("Strassen (threshold = 2)", strassen_ops_2);

    OpCount strassen_ops_64 = countOperations("strassen", 256, 64);
    printOpCount("Strassen (threshold = 64)", strassen_ops_64);

    // Calculate the difference between Strassen and standard multiplication results
    std::cout << "Difference norm: " << (C_strassen - C_standard).norm() << "\n";

    return 0;
}
