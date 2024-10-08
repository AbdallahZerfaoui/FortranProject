#include "solver.hpp"

// Function to simulate 'passage' which converts a 1D index to 2D coordinates (i, j)
void passage(int l, int &i, int &j, int Nx) 
{
    i = (l - 1) / Nx + 1;
    j = (l - 1) % Nx + 1;
}

void populateMatrix(MatrixXd& A, int Nx, int n, double dx, double dy, double D) 
{
	// cout << "Nx: " << Nx << ", n: " << n << ", dx: " << dx << ", dy: " << dy << ", D: " << D << endl;
	// cout << "diag : " << -2.0 / pow(dx, 2) - 2.0 / pow(dy, 2) << endl;
    // Populate the matrix A
    for (int k = 0; k < n; k++) {
        A(k, k) = 2.0 * D * ( 1 / pow(dx, 2) + 1 / pow(dy, 2));  // Diagonal elements

        if (k % Nx != Nx - 1) {
            A(k, k + 1) = -D / pow(dx, 2); 
			A(k + 1, k) = -D / pow(dx, 2); 
        }

        if (k + Nx < n) {
            A(k, k + Nx) = -D / pow(dy, 2);  
			A(k + Nx, k) = -D / pow(dy, 2);
        }
    }
}

void populateVectorF(VectorXd& F, int Nx, int Ny, double dt, double Lx, double Ly) 
{
	int n = Nx * Ny, k = 0; // k is the time step
	double dx = Lx/(Nx + 1), dy = Ly/(Ny + 1);

    // Populating the vector F
    for (int l = 1; l <= n; l++) {
        int i, j;
        passage(l, i, j, Nx);
        F(l - 1) = f1(j * dx, i * dy, (k + 1) * dt);
    }

    // Handling boundaries
    for (int l = 1; l <= Nx; l++) {
        F(l - 1) += g(l * dx, 0.0) / pow(dy, 2);
    }

    for (int l = n - Nx + 1; l <= n; l++) {
        F(l - 1) += g((l - n + Nx) * dx, Ly) / pow(dy, 2);
    }

	for (int l = 1; l <= n - Nx + 1; l += Nx) {
        int i0, j0;
        passage(l, i0, j0, Nx);
        F(l - 1) += h(0.0, i0 * dy) / pow(dx, 2);
    }

    for (int l = Nx; l <= n; l += Nx) {
        int i0, j0;
        passage(l, i0, j0, Nx);
        F(l - 1) += h(Lx, i0 * dy) / pow(dx, 2);
    }

    // Special cases for F(1) and F(n) (adjusted for 0-based indexing)
    // F(0) += h(0.0, dy) / (dx * dx);
    // F(n - 1) += h(0.0, (n - 1) * dy) / (dx * dx);
}

void displayMatrix(const MatrixXd& A) 
{
	cout << "Matrix A:" << endl;
	cout << A << endl;
}

void displayVector(const VectorXd& V) 
{
	cout << "Vector V:" << endl;
	cout << V << endl;
}


int main() 
{
    // int Nx = 50, Ny = 50;  // Example size for the matrix
	auto [Nx, Ny, Lx, Ly, D] = read_parameters("./data");
    int n = Nx * Ny, i0, j0;

    double dx = Lx/(Nx + 1), dy = Ly/(Ny + 1), dt = 1.0;

    // Matrix and vector allocation using Eigen
    MatrixXd A = MatrixXd::Zero(n, n);  // zero matrix
    VectorXd U = VectorXd::Zero(n);     // zero vector
    VectorXd F = VectorXd::Zero(n);       // Result vector

	// Populate the matrix A
	populateMatrix(A, Nx, n, dx, dy, D);
	// displayMatrix(A);

	// Populate the vector F
	populateVectorF(F, Nx, Ny, dt, Lx, Ly);
	// displayVector(F);

    // Solve the system using conjugate gradient method
    conjugate_grad(A, U, F, n, Nx);

    // Save results to files
    ofstream file_vectU_b("VecteurU_b"), file_vectU("VecteurU");

    for (int l = 1; l <= n; ++l) {
        file_vectU << U(l - 1) << endl;  // Save the vector U to file

        passage(l, i0, j0, Nx);
        file_vectU_b << j0 * dx << " " << i0 * dy << " " << U(l - 1) << endl;  // Save the coordinates and values
    }

    file_vectU.close();
    file_vectU_b.close();

	// displayVector(U);

    cout << "Computation complete, results saved to files." << endl;

}
