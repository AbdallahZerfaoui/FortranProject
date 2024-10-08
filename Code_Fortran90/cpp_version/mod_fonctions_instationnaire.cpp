#include "solver.hpp"

// Global variables to store parameters from the file
int Nx, Ny;
double Lx, Ly, D;

// Function to read parameters from a file
array<double,NBR_PARAMS> read_parameters(const string& filename) {
    array<double,NBR_PARAMS> params;
	ifstream file(filename);
    if (file.is_open()) {
        file >> params[0] >> params[1];   // Read Nx and Ny
        file >> params[2] >> params[3];   // Read Lx and Ly
        file >> params[4];          // Read D
        file.close();
    } else {
        cerr << "Unable to open file." << endl;
    }
	return params;
}

// Function f1 - uses x, y, t, and parameters from the file
double f1(double x, double y, double t) 
{
    // Assuming this is how you want to calculate f1 based on Fortran code
    return 2 * ((x - pow(x, 2)) + (y - pow(y, 2))); 
	// return exp(-pow(x-Lx/2, 2)) * exp(-pow(y-Ly/2, 2)) * cos(PI * t / 2);
}

// Function g - simple operation using x and y
double g(double x, double y) 
{
    //return sin(x) + cos(y);  // A simple combination of trigonometric functions
	return 0.0;
}

// Function h - another operation, always returns 1 in this example
double h(double x, double y) 
{
    return 0.0;  // This was hardcoded in Fortran as h = 1.d0
}

