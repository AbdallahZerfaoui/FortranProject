#include "include/GridParameters.hpp"
#include <iostream>
#include <fstream>
#include <string>

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
	int Nx, Ny;
	double Lx, Ly, D;

	auto result = loadConfigData("data");
	
	std::tie(Nx, Ny, Lx, Ly, D) = result;
	GridParameters Grid = GridParameters(Nx, Ny, Lx, Ly, D);

	std::cout << Nx << " " << Ny << " " << Lx << " " << Ly << " " << D << std::endl;
}