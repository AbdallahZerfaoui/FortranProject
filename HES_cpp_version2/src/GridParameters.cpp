#include "../include/GridParameters.hpp"
#include <iostream>
#include <fstream>
#include <string>

// Default constructor
GridParameters::GridParameters(int Nx, int Ny, double Lx, double Ly, double D)
{
	_Nx = Nx; // Number of grid points in x-direction
	_Ny = Ny; // Number of grid points in y-direction
	_Lx = Lx; // Length of the domain in x-direction
	_Ly = Ly; // Length of the domain in y-direction
	_D = D;   // Diffusion coefficient
	// Initialize other member variables
	_n = Nx * Ny; // Total number of grid points
	_dx = Lx / (Nx + 1); // Grid spacing in x-direction
	_dy = Ly / (Ny + 1); // Grid spacing in y-direction
}

// Copy constructor
GridParameters::GridParameters(const GridParameters& other)
{
	// std::cout << "GridParameters copy constructor called" << std::endl;
	// Copy member variables from 'other'.
	// Often done by calling the copy assignment operator:
    *this = other;
    // Alternatively, copy them directly here:
    // 
	// Copy member variables
	_n = other._n;
	_Nx = other._Nx;
	_Ny = other._Ny;
	_Lx = other._Lx;
	_Ly = other._Ly;
	_D = other._D;
	_dx = other._dx;
	_dy = other._dy;
}

// Copy assignment operator
GridParameters& GridParameters::operator=(const GridParameters& other)
{
	// std::cout << "GridParameters copy assignment operator called" << std::endl;
	if (this != &other)
	{   
	// Copy member variables
		this->_n = other._n;
		this->_Nx = other._Nx;
		this->_Ny = other._Ny;
		this->_Lx = other._Lx;
		this->_Ly = other._Ly;
		this->_D = other._D;
		this->_dx = other._dx;
		this->_dy = other._dy;
		// Assign member variables from 'other' (assigned above or add more manually)
	}
	return *this;
}

// Destructor
GridParameters::~GridParameters()
{
	std::cout << "GridParameters destructor called" << std::endl;
}

// void GridParameters::globalIndexToGrid(int k, int& i, int& j)
// {
// 	// std::cout << "GridParameters globalIndexToGrid called" << std::endl;
// 	// Convert global index 'k' to grid indices 'i' and 'j'
// 	// Assuming a 2D grid, you can calculate 'i' and 'j' as follows:
// 	i = k / Ny; // Row index
// 	j = k % Ny; // Column index
// 	// Adjust if necessary based on your grid layout
// }
// int GridParameters::gridIndexToGlobal(int i, int j)
// {
// 	// std::cout << "GridParameters gridIndexToGlobal called" << std::endl;
// 	// Convert grid indices 'i' and 'j' to a global index 'k'
// 	// Assuming a 2D grid, you can calculate 'k' as follows:
// 	return i * Ny + j; // Global index
// 	// Adjust if necessary based on your grid layout
// }


