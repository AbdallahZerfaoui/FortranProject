#include "include/GridParameters.hpp"
#include "include/Vectors.hpp"
#include "include/settings.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

std::tuple<int, int, double, double, double> loadConfigData(std::string filename)
{
	int Nx, Ny;
	double Lx, Ly;
	double D;

	// first we read the file data to extract the parameters
	std::ifstream inputFile(filename);
	if (!inputFile.is_open()) 
	{
		std::cerr << "Error opening file" << std::endl;
		return {0, 0, 0.0, 0.0, 0.0};
	}
	inputFile >> Nx >> Ny;
	inputFile >> Lx >> Ly;
	inputFile >> D;
	inputFile.close();
	return {Nx, Ny, Lx, Ly, D};
}

int main()
{
	std::cout << GREEN<<"1. ---load config data---" << RESET<<std::endl;
	int n;
	int Nx, Ny;
	double Lx, Ly, D;

	auto result = loadConfigData("data");
	
	std::tie(Nx, Ny, Lx, Ly, D) = result;

	std::cout << GREEN<<"2. ---created GridParameters ---" << RESET<<std::endl;
	GridParameters Grid = GridParameters(Nx, Ny, Lx, Ly, D);

	std::cout << Nx << " " << Ny << " " << Lx << " " << Ly << " " << D << std::endl;

	std::cout << GREEN<<"3. ---test SequentialVector Class ---" << RESET<<std::endl;
	// Create two sequential vectors of size n
	// Use unique_ptr to manage memory and interact via base class pointer (optional but good practice)
	n = Nx * Ny; // Total number of elements in the grid
	SequentialVector u(n);
	SequentialVector v(n);
	u.ones(); // Initialize u with ones
	v.ones(); // Initialize v with ones
	double product = 0.0;
	product = u.dot(v);

	std::cout << "Dot product of u and v: " << product << std::endl;
	return 0;
}