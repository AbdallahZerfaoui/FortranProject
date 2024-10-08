#include <iostream>
#include <vector>
#include <cmath>

// Function for matrix-vector multiplication (like prodMV)
std::vector<double> prodMV(const std::vector<std::vector<double>>& A, 
							const std::vector<double>& U, 
							int n, int Nx) 
{
    std::vector<double> result(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i] += A[i][j] * U[j];
        }
    }
    return result;
}

// Subroutine Grad_conjuge equivalent in C++
void Grad_conjuge(const std::vector<std::vector<double>>& A, std::vector<double>& U, const std::vector<double>& F, int n, int Nx) {
    std::vector<double> Gr(n), DIR(n), V(n), Uk(n), Grk(n);
    double alpha, p, epsi = 1E-9;
    int compt, l;

    // Initialize Gr = A * U - F
    Gr = prodMV(A, U, n, Nx);
    for (int i = 0; i < n; ++i) {
        Gr[i] -= F[i];
    }

    // Initialize DIR = -Gr
    for (int i = 0; i < n; ++i) {
        DIR[i] = -Gr[i];
    }

    // Conjugate gradient loop
    compt = 0;
    while (compt < n) {
        // V = A * DIR
        V = prodMV(A, DIR, n, Nx);

        // p = DIR^T * V
        p = 0.0;
        for (int i = 0; i < n; ++i) {
            p += DIR[i] * V[i];
        }

        // alpha = -(Gr^T * DIR) / p
        alpha = 0.0;
        for (int i = 0; i < n; ++i) {
            alpha += Gr[i] * DIR[i];
        }
        alpha = -alpha / p;

        // Update U: U = U + alpha * DIR
        for (int i = 0; i < n; ++i) {
            U[i] += alpha * DIR[i];
        }

        // Update Grk: Grk = Gr + alpha * V
        for (int i = 0; i < n; ++i) {
            Grk[i] = Gr[i] + alpha * V[i];
        }

        // Check convergence
        double norm_Grk = 0.0;
        for (int i = 0; i < n; ++i) {
            norm_Grk += Grk[i] * Grk[i];
        }
        if (sqrt(norm_Grk) < epsi) {
            break;
        }

        // Update direction DIR: DIR = -Grk + ((Grk^T * Grk) / (Gr^T * Gr)) * DIR
        double beta = 0.0, norm_Gr = 0.0;
        for (int i = 0; i < n; ++i) {
            beta += Grk[i] * Grk[i];
            norm_Gr += Gr[i] * Gr[i];
        }
        beta /= norm_Gr;

        for (int i = 0; i < n; ++i) {
            DIR[i] = -Grk[i] + beta * DIR[i];
        }

        // Update Gr
        Gr = Grk;

        compt++;
    }
}

int main() {
    // Example usage
    int n = 5, Nx = 5;
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0)); // Example matrix A
    std::vector<double> U(n, 1.0);  // Initial guess for U
    std::vector<double> F(n, 1.0);  // Right-hand side vector F

    // Initialize A with some values (example)
    for (int i = 0; i < n; ++i) {
        A[i][i] = 2.0;
        if (i < n - 1) {
            A[i][i + 1] = -1.0;
            A[i + 1][i] = -1.0;
        }
    }

    // Call the conjugate gradient method
    Grad_conjuge(A, U, F, n, Nx);

    // Output the result
    std::cout << "Solution U: ";
    for (double u : U) {
        std::cout << u << " ";
    }
    std::cout << std::endl;

    return 0;
}
