#include <iostream>
#include <iomanip>
#include <cstddef>
#include <string>
#include <cstdlib>
#include <random>

#include <mkl.h>

#define N_CM(i, j, M, N) (i + j*M)

#define PRINT_MATRIX(matrix, m, n) printColMajor(#matrix, matrix, m, n)

void printColMajor(const std::string& matrixName,
                   const double* matrix, std::size_t m, std::size_t n)
{
    std::cout << matrixName << " = {\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::setw(6) << matrix[N_CM(i,j, m, n)];
        }
        std::cout << '\n';
    }
    std::cout << "}\n";
}

void smallMatrices()
{
    const int m = 3;
    const int k = 3;
    const int n = 3;

    std::cout << "--- Aligned matrices\n"
                 "Allocating memory for matrices aligned on 64-byte boundary "
                 "for better performance\n";

    double* ama; // m x k
    double* amb; // k x n
    double* amc; // m x n

    ama = static_cast<double*>(mkl_malloc(m*k*sizeof(double), 64));
    amb = static_cast<double*>(mkl_malloc(k*n*sizeof(double), 64));
    amc = static_cast<double*>(mkl_malloc(m*n*sizeof(double), 64));

    if (ama == nullptr || amb == nullptr || amc == nullptr) {
        std::cerr << "error: cannot allocate memory for matrices. Aborting...";
        mkl_free(ama);
        mkl_free(amb);
        mkl_free(amc);
        return;
    }

    std::cout << "Initializing matrix data\n";

    for(int i = 0; i < m*k; ++i) {
        if(i < 3) {
            ama[i] = 1;
        } else if (i < 6) {
            ama[i] = 2;
        } else {
            ama[i] = 3;
        }
    }

    for(int i = 0; i < k*n; ++i) {
        if(i > 3 && i < 7) {
            amb[i] = 0;
        } else {
            amb[i] = 1;
        }
    }

    for (int i = 0; i < m*n; ++i) {
        amc[i] = 0;
    }

    PRINT_MATRIX(ama, m, k);
    PRINT_MATRIX(amb, k, n);

    std::cout << "\nComputing matrix product using Intel(R) MKL dgemm function "
                 "via CBLAS interface\n\n";

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
        1, ama, m, amb, k, 1, amc, n);

    PRINT_MATRIX(amc, m, n);

    mkl_free(ama);
    mkl_free(amb);
    mkl_free(amc);

    std::cout << "\n--- Built-in matrices\n";
    double ma[] = {1, 1, 1, 2, 2, 2, 3, 3, 3};
    double mb[] = {1, 1, 1, 1, 0, 0, 0, 1, 1};
    double mc[9] = {};

    PRINT_MATRIX(ma, m, k);
    PRINT_MATRIX(mb, k, n);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                1, ma, m, mb, k, 1, mc, n);

    PRINT_MATRIX(mc, m, n);
}

int main()
{
    double *aA, *aB, *aC;

    const int m = 25'000;
    const int k = 5'000;
    const int n = 20'000;

    const double alpha = 1.0;
    const double beta = 1.0;

    std::cout << "Allocating memory for matrices aligned on 64-byte boundary\n";
    aA = static_cast<double*>(mkl_malloc(m*k*sizeof(double), 64));
    aB = static_cast<double*>(mkl_malloc(k*n*sizeof(double), 64));
    aC = static_cast<double*>(mkl_malloc(m*n*sizeof(double), 64));

    if (aA == nullptr || aB == nullptr || aC == nullptr) {
        std::cerr << "error: cannot allocate memory for matrices. Aborting...";
        mkl_free(aA);
        mkl_free(aB);
        mkl_free(aC);
        return EXIT_FAILURE;
    }

    std::cout << "Initializing matrix data\n";
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> uniDist(0.0, 1.0);

    for (int i = 0; i < m*k; ++i) {
        aA[i] = uniDist(rng);
    }

    std::cout << "Initialized matrix A\n";

    for (int i = 0; i < k*n; ++i) {
        aB[i] = uniDist(rng);
    }

    std::cout << "Initialized matrix B\n";

    for (int i = 0; i < m*n; ++i) {
        aC[i] = 0;
    }

    std::cout << "Initialized matrix C\n";

    std::cout << "Computing matrix product...\n";
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
        alpha, aA, k, aB, n, beta, aC, n);

    std::cout << "Finished computing!";

    mkl_free(aA);
    mkl_free(aB);
    mkl_free(aC);

    return EXIT_SUCCESS;
}