#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <iostream>
#include <Eigen/Dense> // Include Eigen for matrix and vector operations
#include <fstream>
#include <cmath> // for sin, cos, acos
#include <array> // for array

using namespace std;
using namespace Eigen;

// Constants
#define PI acos(-1.0)
#define EPSI 1E-6
#define NBR_PARAMS 5

//Functions
double f1(double x, double y, double t);
double g(double x, double y);
double h(double x, double y);

array<double,NBR_PARAMS> read_parameters(const string& filename);
void conjugate_grad(const Eigen::MatrixXd& A, Eigen::VectorXd& U, const Eigen::VectorXd& F, int n, int Nx);
void passage(int l, int &i, int &j, int Nx);
void populateMatrix(MatrixXd& A, int Nx, int n, double dx, double dy, double D);


#endif