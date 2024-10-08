#include "solver.hpp"

// Function conjugate_grad equivalent in C++
void conjugate_grad(const Eigen::MatrixXd& A, Eigen::VectorXd& U, const Eigen::VectorXd& F, int n, int Nx) 
{
    Eigen::VectorXd Gr(n), DIR(n), V(n), Grk(n);
    double alpha, p, epsi = 1E-9;
    int compt;

    // Initialize Gr = A * U - F
    Gr = A * U - F;

    // Initialize DIR = -Gr
    DIR = -Gr;

    // Conjugate gradient loop
    compt = 0;
    while (Gr.norm()/F.norm() > EPSI) {
		cout << compt << " Gr.norm()/F.norm() = " << Gr.norm()/F.norm() << endl;
        // V = A * DIR
        V = A * DIR;

        // p = DIR^T * V
        // p = DIR.dot(V);

        // alpha = -(Gr^T * DIR) / DIR.dot(V)
        alpha = Gr.squaredNorm() / DIR.dot(V);

        // Update U: U = U + alpha * DIR
        U = U + alpha * DIR;

        // Update Grk: Grk = Gr + alpha * V
		Grk = Gr;
        Gr = Gr + alpha * V;

        // Check convergence using the norm of Grk
        // if (Grk.norm() < epsi) {
        //     break;
        // }

        // Update direction DIR: DIR = -Grk + ((Grk^T * Grk) / (Gr^T * Gr)) * DIR
        p = Gr.squaredNorm() / Grk.squaredNorm();
        DIR = -Gr + p * DIR;

        // Update Gr
        // Gr = Grk;

        compt++;
    }
}

// int main() {
//     // Example usage
//     int n = 5, Nx = 5;
    
//     // Define a matrix A using eigen::MatrixXd
//     Eigen::MatrixXd A(n, n);  // Matrix A of size n x n

//     // Define vectors U and F using eigen::VectorXd
//     Eigen::VectorXd U(n);     // Solution vector U of size n
//     Eigen::VectorXd F(n);     // Right-hand side vector F of size n

//     // Initialize A with some example values
//     A.setZero(); // Set all elements to zero initially
//     for (int i = 0; i < n; ++i) {
//         A(i, i) = 2.0;         // Diagonal elements set to 2.0
//         if (i < n - 1) {
//             A(i, i + 1) = -1.0; // Upper off-diagonal
//             A(i + 1, i) = -1.0; // Lower off-diagonal
//         }
//     }

//     // Initialize U and F with some values
//     U.setOnes(); // Initialize U with ones
//     F.setOnes(); // Initialize F with ones

//     // Call the conjugate gradient method
//     conjugate_grad(A, U, F, n, Nx);

//     // Output the result
//     std::cout << "Solution U: " << U.transpose() << std::endl;

//     return 0;
// }
